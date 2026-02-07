from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.checkpoint.postgres import PostgresSaver
import os
import psycopg


load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")

connection = psycopg.connect(DB_URL, sslmode="require", autocommit=True)


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.5, max_retries=2)


sys_prompt = SystemMessage(
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

**Typography (MANDATORY)**
* You MUST use **professional Google Fonts** only (e.g., Montserrat, Playfair Display, Poppins, Oswald, Lato, Roboto, etc. ).
* Import fonts using `@import` inside the `<style>` block.
* Apply **type hierarchy**.

**Imagery Rules**
* Use stock images ONLY when realism is essential.
* Otherwise, create **abstract, geometric, or gradient-based compositions**.

---

### 🖼 IMAGE SOURCE RULES (If Needed)

If a stock image is required:
1. Use `https://picsum.photos/seed/{keyword}/{width}/{height}`
2. Replace `{keyword}` with a specific subject.
3. **MANDATORY:** You must set `img.crossOrigin = "anonymous";` before setting the `src`.

---

### 📦 OUTPUT FORMAT (STRICT)

You must respond with **ONLY valid JSON** — no markdown, no extra text.
The JSON must contain **exactly three keys**:
{
  "ai_message": "Short explanation of design choices.",
  "canvas": "Full standalone HTML document as a string",
  "title": "A short name for the design generated"
}

---

### ⚙️ CODE RULES ("canvas" value)
• No external CSS files.
• Canvas default size: **1800 × 2400 vertical**.
• You MUST include a **text-wrapping helper function**.
• Add these styles to the `<canvas>` element: `max-width: 100%; max-height: 100%; object-fit: contain;`.

### ⏳ FONT & IMAGE LOADING LOGIC (MANDATORY)
1. Import fonts via CSS.
2. Use `document.fonts.ready.then(() => { ... });`.
3. Load images inside the font promise.
4. Draw only inside `img.onload`.

If something feels crowded, fix it.

Finally add this properties on the canvass element generated

max-width: 100%;  /* Shrink to fit width */
max-height: 100%; /* Shrink to fit height */
object-fit: contain; /* Keeps the aspect ratio perfect */
"""
)


# Setting Message state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Setting up nodes
def chat_node(state: State):
    messages = [sys_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Building Graph

builder = StateGraph(State)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

memory = PostgresSaver(connection)
memory.setup()
graph = builder.compile(checkpointer=memory)


# Function to chat with agents


def chat_with_agent(user_id: str, session_id: str, message: str):
    """
    Each user has a peculiar id which includes:
    1. Their personal user_id
    2. session_id generated per new chat (which is used to retrieve previous chat)
    This recieves input from the backend and sends back AI response

    """
    try:
        unique_id = f"{user_id}: {session_id}"
        config = {"configurable": {"thread_id": unique_id}}
        human_msg = HumanMessage(content=message)
        reply = graph.invoke({"messages": [human_msg]}, config=config)

        last_msg = reply["messages"][-1]

        content = last_msg.content
        if isinstance(content, list):
            text = "".join(p["text"] for p in content if "text" in p)

        else:
            text = str(content)

        return text

    except Exception as e:
        return f"Error : {str(e)}"


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
        unique_id = f"{user_id}: {session_id}"
        config = {"configurable": {"thread_id": unique_id}}

        # Get state from graph
        state = graph.get_state(config)

        # Extract messages
        conversation = []
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    text = "".join(p["text"] for p in content if "text" in p)
                else:
                    text = str(content)

                conversation.append({"role": "ai", "content": text})

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
        cursor.execute(
            """
            SELECT DISTINCT thread_id 
            FROM checkpoints 
            WHERE thread_id LIKE %s
        """,
            (f"{user_id}:%",),
        )

        results = cursor.fetchall()

        sessions = []
        for row in results:
            thread_id = row[0]  # e.g., "user123: session456"

            # Extract session_id
            session_id = thread_id.split(": ")[1]

            # Get first message as preview
            history = get_conversation_history(user_id, session_id)
            preview = "New chat"
            if history and len(history) > 0:
                preview = history[0]["content"][:50]  # First 50 characters

            sessions.append({"session_id": session_id, "preview": preview})

        cursor.close()
        return sessions

    except Exception as e:
        return []
