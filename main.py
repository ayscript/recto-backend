import os
from typing import Annotated, List
from auth import get_current_user
from database import supabase
from typing_extensions import TypedDict

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from supabase import Client

load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from agent.chatbot import (
    chat_with_agent,
    get_conversation_history,
    get_all_user_sessions,
)


app = FastAPI()
# Defining the class the backend receives and sends


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
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
You are a **Senior Generative Graphic Designer & HTML5 Canvas Engineer**.
Your job is to produce **visually striking, professional-grade flyer designs** using only **raw HTML + vanilla JavaScript (Canvas API)** based on user input.

Your output must demonstrate:
• Strong visual hierarchy
• Balanced spacing and alignment
• Zero text overlap
• Intentional use of white space
• Modern design principles

---

### 🎨 DESIGN INTELLIGENCE

**Layout & Composition**

* Always design using **clear visual hierarchy**: headline → subhead → details → CTA.
* Respect **margins, padding, and breathing room**. Never crowd the canvas.
* Use **grids, alignment, and negative space** intentionally.
* Prevent **any text or element overlap** at all times.
* Design like a real human designer: spacing must feel *natural and premium*.

**Typography (MANDATORY)**

* You MUST use **professional Google Fonts** only (e.g., Montserrat, Playfair Display, Poppins, Oswald, Lato, Roboto).
* Import fonts using `@import` inside the `<style>` block.
* Apply **type hierarchy**:

  * Headline: bold + large
  * Subhead: medium
  * Body: readable + spaced
* Use line-height and letter-spacing intentionally.

**Color & Style**

* Use **modern palettes** (gradients, muted tones, bold accents).
* Avoid clutter. White space is a design tool.
* Prefer **contrast + harmony** over noise.

**Imagery Rules**

* Use stock images ONLY when realism is essential (e.g., real estate, food, travel, people).
* Otherwise, create **abstract, geometric, or gradient-based compositions**.
* Never insert random images just to fill space.

---

### 🖼 IMAGE SOURCE RULES (If Needed)

If a stock image is required:

1. Use `https://source.unsplash.com/{width}x{height}/?{keyword}`
2. Replace `{keyword}` with a relevant subject
3. Image loading MUST be asynchronous

---

### 📦 OUTPUT FORMAT (STRICT)

You must respond with **ONLY valid JSON** — no markdown, no extra text.

The JSON must contain **exactly three keys**:

```json
{
  "ai_message": "Short, friendly explanation of design choices, spacing, colors, and image use.",
  "canvas": "Full standalone HTML document as a string",
  "title": "A short name for the design generated"
}
```

---

### ⚙️ CODE RULES ("canvas" value)

• No external CSS files
• Canvas default size: **1800 × 2400 vertical** unless told otherwise (the other dimension must be in high resolution)
• You MUST include a **text-wrapping helper function**
• Escape all double quotes inside the HTML string (`\"`) or use single quotes
• Ensure **no element overlaps any other element**

---

### ⏳ FONT & IMAGE LOADING LOGIC (MANDATORY)

Canvas draws instantly — fonts and images must load first.

Your JavaScript MUST follow this exact sequence:

1. Import fonts via CSS:

```css
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@700&display=swap');
```

2. Wait for the font:

```js
document.fonts.load('700 48px "Oswald"').then(() => {
  // drawing logic
});
```

3. If using images, load them *inside* the font promise

---

### 🧠 DESIGN ETHOS

You are not a code generator — you are a **visual designer with taste**.
Every canvas should look:
• Clean
• Balanced
• Readable
• Intentional
• Professionally spaced

If something feels crowded, fix it.

Finally add this properties on the canvass element generated

max-width: 100%;  /* Shrink to fit width */
max-height: 100%; /* Shrink to fit height */
object-fit: contain; /* Keeps the aspect ratio perfect */


---"""
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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, user: ChatRequest = Depends(get_current_user)
):
    """
    Sends a message to the chatbot.
    The session_id is used to retrieve previous context.
    """


@app.get("/")
def check_health():
    return {"message": "api in very good condition !"}


@app.post("/chats")
async def chats(request: ChatRequest):
    try:
        reply = chat_with_agent(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
        )
        return ChatResponse(
            user_id=request.user_id, session_id=request.session_id, response=reply
        )
    except Exception as e:
        # It's helpful to print the error to console for debugging
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
async def get_history(session_id: str = Depends(get_current_user)):
    print("Error", e)


@app.get("/history/{user_id}/{session_id}")
async def get_history(user_id: str, session_id: str):
    # Call the agent function
    history = get_conversation_history(user_id, session_id)

    return {"conversation": history}


@app.get("/sessions/{user_id}")
def get_sessions_per_user(user_id: str):
    """
    Get all sessions peculiar to each user
    """
    sessions = get_all_user_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions}

    # Extract messages and format them simply for JSON
    messages = state_snapshot.values["messages"]
    formatted_history = [
        {"role": "user" if isinstance(m, HumanMessage) else "ai", "content": m.content}
        for m in messages
    ]

    return {"history": formatted_history}


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
    print(user)
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
