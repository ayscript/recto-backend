from typing import Annotated
from auth import get_current_user
from database import supabase
from typing_extensions import TypedDict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from agent.chatbot import (
    chat_with_agent,
    get_conversation_history,
    get_all_user_sessions,
)


app = FastAPI()
# Defining the class the backend receives and sends


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    response: str


# --- 3. FASTAPI APP ---

app = FastAPI(title="Recto AI Backend")

origins = [
    "http://localhost:3000",  # React default port
    "http://localhost:5173",  # React default port
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allowed domains
    allow_credentials=True,  # Allow cookies/auth headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/health")
def health_check():
    return {"message": "System is in good condition"}


@app.post("/chat")
async def chats(request: ChatRequest, user=Depends(get_current_user)):
    try:
        reply = chat_with_agent(
            user_id=user.user.id,
            session_id=request.session_id,
            message=request.message,
        )
        return ChatResponse(
            user_id=user.user.id, session_id=request.session_id, response=reply
        )
    except Exception as e:
        # It's helpful to print the error to console for debugging
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/history/{session_id}")
async def get_history(session_id: str, user=Depends(get_current_user)):
    user_id = user.user.id
    # Call the agent function
    history = get_conversation_history(user_id, session_id)

    return {"conversation": history}


@app.get("/sessions/")
def get_sessions_per_user(user=Depends(get_current_user)):
    """
    Get all sessions peculiar to each user
    """
    user_id = user.user.id
    sessions = get_all_user_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions}

    # Extract messages and format them simply for JSON
    # messages = state_snapshot.values["messages"]
    # formatted_history = [
    #     {"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
    #     for m in messages
    # ]

    # return {"history": formatted_history}


# Authentication


class SignupSchema(BaseModel):
    """Schema for the user signup request body."""

    email: EmailStr
    password: str
    display_name: str


class LoginSchema(BaseModel):
    """Schema for the user signup request body."""

    email: EmailStr
    password: str


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
        res = supabase.auth.sign_in_with_password(
            {"email": payload.email, "password": payload.password}
        )
        return {"access_token": res.session.access_token, "token_type": "bearer"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/get_profile")
def get_user_details(user=Depends(get_current_user)):
    # 'user' is the object returned by supabase.auth.get_user(token)
    
    return {
        "id": user.user.id,
        "email": user.user.email,
        "created_at": user.user.created_at,
        "last_sign_in": user.user.last_sign_in_at,
        "metadata": user.user.user_metadata,  # Contains custom fields like 'display_name'
    }


# --- 4. RUNNER (For debugging) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
