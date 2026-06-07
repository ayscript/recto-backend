import time
from typing import Annotated, List, Optional
from fastapi import Depends, FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

# Import dependencies
from auth import get_current_user
from database import supabase
from agent.chatbot import (
    chat_with_agent,
    get_conversation_history,
    get_all_user_sessions,
    connection,
)

load_dotenv()

# --- 1. SCHEMAS ---

# Kept schema definitions intact for schema validation safety
class AssetDescriptionItem(BaseModel):
    file_name: str
    description: Optional[str] = ""
    public_url: Optional[str] = ""

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default-playground"  # 
    asset_descriptions: Optional[List[AssetDescriptionItem]] = []
    
class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str

class SignupSchema(BaseModel):
    email: EmailStr
    password: str
    display_name: str

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

# --- 2. FASTAPI APP SETUP ---

app = FastAPI(title="Recto AI Backend")

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "https://recto-app.netlify.app",
    "https://recto.pxxl.click",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. CHAT & ROUTE ALIGNMENT ---

@app.get("/health")
def health_check():
    return {"message": "System is in good condition"}

@app.post("/chat")
async def chats(request: ChatRequest, user = Depends(get_current_user)):
    try:
        user_id = user.user.id
        session_id = request.session_id
        
        # ⚡ SIMPLIFIED: No longer tracking asset DB records. 
        # The frontend injects HTML img tags or context straight into request.message.
        reply = chat_with_agent(
            user_id=user_id,
            session_id=session_id,
            message=request.message
        )
        
        return {"response": reply}
        
    except Exception as e:
        print(f"Chat Route Crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_history(session_id: str, user=Depends(get_current_user)):
    try:
        user_id = user.user.id
        history = get_conversation_history(user_id, session_id)
        
        if not history:
            return {"conversation": []}
            
        return {"conversation": history}
    except Exception as e:
        print(f"History Fetch Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/session/init")
async def initialize_draft_session(user = Depends(get_current_user)):
    try:
        import uuid
        generated_session_id = str(uuid.uuid4())
        return {"session_id": generated_session_id, "status": "draft"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize workspace transaction: {str(e)}")

@app.get("/sessions")
def get_sessions_per_user(user=Depends(get_current_user)):
    user_id = user.user.id
    sessions = get_all_user_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions}

@app.delete("/session/delete/{session_id}")
async def delete_session(session_id: str, user=Depends(get_current_user)):
    try:
        user_id = user.user.id
        unique_id = f"{user_id}: {session_id}"
        cursor = connection.cursor()
        cursor.execute(
            """
            DELETE FROM checkpoints
            WHERE thread_id = %s
            """,
            (unique_id,)
        )
        cursor.close()
        return {"message": "Session deleted", "session_id": session_id}
    except Exception as e:
        print(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 4. ASSET MANAGEMENT ENDPOINTS ---

@app.post("/upload-asset")
async def upload_asset(
    file: UploadFile = File(...),
    assetType: str = Form(...),
    session_id: str = Form(...),  
    user = Depends(get_current_user)
):
    try:
        user_id = user.user.id
        
        # 1. Process file streams
        file_bytes = await file.read()
        
        # 2. Structure paths following your dynamic folder architecture: users/{user_id}/{filename}
        sanitized_filename = f"{int(time.time())}_{file.filename}"
        storage_path = f"users/{user_id}/{sanitized_filename}"

        # 3. Stream binary directly into the Supabase bucket
        supabase.storage.from_("design-assets").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": file.content_type, "upsert": "true"}
        )

        # 4. Extract public accessible source string 
        public_url_res = supabase.storage.from_("design-assets").get_public_url(storage_path)
        public_url = public_url_res if isinstance(public_url_res, str) else getattr(public_url_res, "public_url", "")
        
        # 5. Skip database logging completely. Just return data back to frontend UI state wrapper.
        return {
            "message": "Asset uploaded successfully", 
            "publicUrl": public_url,
            "asset": {
                "file_name": file.filename,
                "publicUrl": public_url
            }
        }

    except Exception as e:
        print(f"File upload processing crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. AUTHENTICATION ENDPOINTS ---

@app.post("/signup")
def signup(payload: SignupSchema):
    try:
        res = supabase.auth.sign_up(
            {
                "email": payload.email,
                "password": payload.password,
                "options": {"data": {"display_name": payload.display_name}},
            }
        )
        if res.user is None:
            raise HTTPException(status_code=400, detail="Signup failed")
        return {"message": "User created", "user": res.user}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail=str(e))

@app.post("/login")
def login(payload: LoginSchema):
    try:
        res = supabase.auth.sign_with_password(  # Use sign_in_with_password if using older supabase-py versions
            {"email": payload.email, "password": payload.password}
        )
        return {"access_token": res.session.access_token, "token_type": "bearer"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail=str(e))

@app.get("/get_profile")
def get_user_details(user=Depends(get_current_user)):
    return {
        "id": user.user.id,
        "email": user.user.email,
        "created_at": user.user.created_at,
        "last_sign_in": user.user.last_sign_in_at,
        "metadata": user.user.user_metadata,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)