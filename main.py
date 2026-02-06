from fastapi import FastAPI
from pydantic import BaseModel
from agent.chatbot import chat_with_agent, get_conversation_history, get_all_user_sessions


app = FastAPI()
# Defining the class the backend receives and sends

class ChatRequest(BaseModel):
    user_id: str
    session_id : str
    message : str
    
class ChatResponse(BaseModel):
    user_id : str
    session_id : str
    response : str
    

@app.get('/')
def check_health():
    return {
        'message' : 'api in very good condition !'
    }
@app.post('/chats')   
async def chats(request : ChatRequest):
    try:
        reply = chat_with_agent(
            user_id=request.user_id,
            session_id= request.session_id,
            message= request.message,
            
        )
        return ChatResponse(
            user_id= request.user_id,
            session_id= request.session_id,
            response = reply
            
        )
    except Exception as e:
        print('Error', e)
@app.get("/history/{user_id}/{session_id}")
async def get_history(user_id: str, session_id: str):
    # Call the agent function
    history = get_conversation_history(user_id, session_id)
    
    return {
        'conversation': history
    }
@app.get('/sessions/{user_id}')
def get_sessions_per_user(user_id : str):
    """
   Get all sessions peculiar to each user
    """
    sessions = get_all_user_sessions(user_id)
    return {
        'user_id': user_id,
        'sessions': sessions
    }
    

    

    

    
