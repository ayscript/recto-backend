import os
from typing import Annotated, List
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION & MODELS ---

# Set your API Key
# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"


# Pydantic models for API Validation
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


# --- 2. LANGGRAPH SETUP ---

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.5,
    max_retries=2,
)

# System Prompt
SYSTEM_PROMPT = SystemMessage(
    content="""
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
}"""
)


# Graph State
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Chat Node
def chatbot(state: State):
    # Prepend the system prompt to the history
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Build Graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# CRITICAL: Add Memory (Checkpointer)
# This allows the graph to persist state based on thread_id
memory = MemorySaver()

# Compile with the checkpointer
graph = builder.compile(checkpointer=memory)

# --- 3. FASTAPI APP ---

app = FastAPI(title="Recto AI Backend")

origins = [
    "http://localhost:3000",      # React default port
    "http://localhost:5173",      # React default port
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allowed domains
    allow_credentials=True,         # Allow cookies/auth headers
    allow_methods=["*"],             # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],             # Allow all headers
)

@app.get("/health")
def health_check():
    return {
        "message": "System is in good condition"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Sends a message to the chatbot.
    The session_id is used to retrieve previous context.
    """
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        input_message = HumanMessage(content=request.message)

        output = graph.invoke({"messages": [input_message]}, config=config)
        last_message = output["messages"][-1]

        # --- FIX STARTS HERE ---
        # content can be a string OR a list of parts (e.g. [{'type': 'text', 'text': ...}])
        raw_content = last_message.content

        if isinstance(raw_content, list):
            # Extract text from all parts and join them
            final_text = "".join(
                [part["text"] for part in raw_content if "text" in part]
            )
        else:
            # It's already a string
            final_text = str(raw_content)
        # --- FIX ENDS HERE ---

        return ChatResponse(response=final_text)

    except Exception as e:
        # It's helpful to print the error to console for debugging
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """
    Retrieves the raw chat history for a specific session.
    """
    config = {"configurable": {"thread_id": session_id}}

    # get_state returns a snapshot of the graph for this config
    state_snapshot = graph.get_state(config)

    if not state_snapshot.values:
        return {"history": []}

    # Extract messages and format them simply for JSON
    messages = state_snapshot.values["messages"]
    formatted_history = [
        {"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in messages
    ]

    return {"history": formatted_history}


# --- 4. RUNNER (For debugging) ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
