from  langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.checkpoint.postgres import PostgresSaver
import os
import psycopg



load_dotenv()

DB_URL= os.getenv('SUPABASE_DB_URL')

connection = psycopg.connect(DB_URL,
                             sslmode="require", autocommit=True)


llm = ChatGoogleGenerativeAI(
    model= 'gemini-2.5-flash',
    temperature= 1.5,
    max_retries =2
)


sys_prompt = SystemMessage(content="""
                           You are an expert Generative Graphic Designer and HTML5 Canvas Specialist. Your goal is to generate high-quality, professional flyer designs using raw HTML and vanilla JavaScript (Canvas API) based on user requests.

### DESIGN PHILOSOPHY
- **Visuals:** Create modern, aesthetically pleasing compositions. Use gradients, geometric masks, blend modes, and thoughtful whitespace.
- **Typography:** **MANDATORY:** You must use professional Google Fonts (e.g., Montserrat, Playfair Display, Roboto, Oswald, Lato) to ensure high-quality design. Import them via `@import` in the `<style>` block.
- **Imagery:** Use stock imagery ONLY when the specific context demands realistic photography (e.g., "Real Estate", "Restaurant", "Travel"). For abstract or corporate themes, rely on algorithmic geometric patterns and gradients.

### IMAGE SOURCE INSTRUCTIONS
If a stock image is strictly necessary:
1. Use the loremflickr Source URL format: `https://loremflickr.com/{width}/{height}/{keyword}`.
2. Replace `{keyword}` with a relevant term (e.g., `pizza`, `house`, `concert`).
3. **CRITICAL:** You must handle image loading asynchronously.

### OUTPUT FORMAT
You must respond strictly in a valid JSON format.
- **NO** markdown formatting (no ```json or ```).
- **NO** conversational text outside the JSON object.
- **NO** trailing commas.

The JSON object must have exactly two keys:
1. "ai_message": A string containing a brief, friendly explanation of the design choices, color palette, and why an image was (or was not) included.
2. "canvas": A string containing the complete, standalone HTML code.

### CODE CONSTRAINTS ("canvas" key)
- **Self-Contained:** No external CSS files.
- **Text Wrapping:** Canvas does not support multi-line text. You MUST write a helper function to wrap text within a specific width.
- **Dimensions:** Default to vertical (e.g., 600x800) unless requested otherwise.
- **Escaping:** The HTML string is inside a JSON value. You MUST escape all double quotes inside the HTML using a backslash (\") or use single quotes (') for HTML attributes.

### CRITICAL: FONT & ASSET LOADING LOGIC
Canvas draws pixels immediately. If the font or image isn't loaded, it draws the wrong thing. You MUST structure your JavaScript exactly like this:

1.  **Define CSS:** `@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&display=swap');` inside `<style>`.
2.  **Wait for Font:** Use `document.fonts.load('700 40px "Oswald"').then(() => { ...Logic... });`
3.  **Wait for Image (Inside Font Promise):** If using an image, load it *inside* the font promise.

### JSON STRUCTURE EXAMPLE
{
  "ai_message": "I used the 'Oswald' Google Font for a bold, impactful headline.",
  "canvas": "<!DOCTYPE html><html><head><style>@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&display=swap'); body { margin: 0; background: #111; display: flex; justify-content: center; }</style></head><body><canvas id='c' width='600' height='800'></canvas><script>const c = document.getElementById('c'); const ctx = c.getContext('2d'); // 1. Wait for Font document.fonts.load('700 60px \"Oswald\"').then(() => { // 2. (Optional) Load Image const img = new Image(); img.crossOrigin='Anonymous'; img.onload = () => { ctx.drawImage(img,0,0); ctx.font = '700 60px \"Oswald\"'; ctx.fillStyle = 'white'; ctx.fillText('TITLE', 50, 100); }; img.src = 'https://loremflickr.com/600/800/city'; });</script></body></html>"
}""")

# Setting Message state
class State(TypedDict):
    messages : Annotated[list, add_messages]
    
# Setting up nodes
def chat_node(state: State):
    messages = [sys_prompt] + state['messages']
    response = llm.invoke(messages)
    return {
        'messages' : [response]
    }
    
# Building Graph

builder = StateGraph(State)
builder.add_node('chat_node', chat_node)
builder.add_edge(START, 'chat_node')
builder.add_edge('chat_node', END)

memory = PostgresSaver(connection)
memory.setup()
graph = builder.compile(checkpointer = memory)




# Function to chat with agents

def chat_with_agent(user_id:str, session_id: str, message: str):
    """
    Each user has a peculiar id which includes:
    1. Their personal user_id
    2. session_id generated per new chat (which is used to retrieve previous chat)
    This recieves input from the backend and sends back AI response
    
    """
    try:
        unique_id = f'{user_id}: {session_id}'
        config = {
            'configurable': {'thread_id' : unique_id}
        }
        human_msg = HumanMessage(content= message)
        reply = graph.invoke(
            {"messages" : [human_msg]},
            config = config
        )
        
        last_msg = reply['messages'][-1]
        
        content = last_msg.content
        if isinstance(content, list):
            text = ''.join(p['text'] for p in content if 'text' in p)
    
        else:
            text = str(content)
            
        return text
    
    except Exception as e:
        return f'Error : {str(e)}'
    
# Add this function at the end of your file

def get_conversation_history(user_id: str, session_id: str):
    """
    Gets the conversation history for a specific user session.
    
    Returns:
        list: [
            {'role': 'user', 'content': '...'},
            {'role': 'ai', 'content': '...'},
            ...
        ]
    """
    try:
        unique_id = f'{user_id}: {session_id}'
        config = {'configurable': {'thread_id': unique_id}}
        
        # Get state from graph
        state = graph.get_state(config)
        
        # Extract messages
        conversation = []
        for msg in state.values.get('messages', []):
            if isinstance(msg, HumanMessage):
                conversation.append({
                    'role': 'user',
                    'content': msg.content
                })
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    text = ''.join(p['text'] for p in content if 'text' in p)
                else:
                    text = str(content)
                
                conversation.append({
                    'role': 'ai',
                    'content': text
                })
        
        return conversation
    
    except Exception as e:
        return []
def get_all_user_sessions(user_id: str):
    """
    Get all chat sessions for a user (for sidebar).
    Returns list of sessions with first message as preview.
    """
    try:
        cursor = connection.cursor()
        
        # Find all conversations for this user
        cursor.execute("""
            SELECT DISTINCT thread_id 
            FROM checkpoints 
            WHERE thread_id LIKE %s
        """, (f'{user_id}:%',))
        
        results = cursor.fetchall()
        
        sessions = []
        for row in results:
            thread_id = row[0]  # e.g., "user123: session456"
            
            # Extract session_id
            session_id = thread_id.split(': ')[1]
            
            # Get first message as preview
            history = get_conversation_history(user_id, session_id)
            preview = "New chat"
            if history and len(history) > 0:
                preview = history[0]['content'][:50]  # First 50 characters
            
            sessions.append({
                'session_id': session_id,
                'preview': preview
            })
        
        cursor.close()
        return sessions
    
    except Exception as e:
        return []