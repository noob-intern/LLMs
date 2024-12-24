import getpass
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# --- ENV SETUP ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter LangChain API Key: ")
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# --- FASTAPI APP ---
app = FastAPI()


# --- Pydantic schema ---
class QueryRequest(BaseModel):
    query: str
    thread_id: str


# --- CREATE YOUR MODEL ---
model = ChatOpenAI(model="gpt-4o-mini")


# --- SETUP THE PROMPT TEMPLATE ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a friendly and knowledgeable customer service representative for Acoustica Guitars, a premium guitar manufacturer known for exceptional craftsmanship and customer care. You are here to assist customers with any questions they may have about the company's products, services, or policies.
            
            Here are the company details: 
            - **Company Name**: MelodyCraft Guitars
            - **Founded**: 1985
            - **Specialties**: Custom electric guitars, acoustic guitars, and basses handcrafted from ethically sourced woods.
            - **Unique Selling Points**:
              - Lifetime warranty on all guitars.
              - A wide range of customization options for body shape, neck profile, and finishes.
              - Exclusive "ToneMaster Pickup Series" for superior sound quality.
            - **Global Presence**: Retail stores in 20 countries and free worldwide shipping.
            - **Customer Perks**: 
              - Free annual guitar maintenance for loyal customers.
              - Access to exclusive online tutorials and masterclasses by renowned musicians.

            When users ask for anything, your goal is to exceed their expectations. Be polite, enthusiastic, and ready to provide detailed assistance about our guitars, services, or policies."""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# --- TRIMMER EXAMPLE (OPTIONAL) ---
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


# --- BUILD THE GRAPH ---
workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    # Trim messages if desired
    trimmed_messages = trimmer.invoke(state["messages"])
    # Build prompt
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    # Call the model
    response = model.invoke(prompt)
    # Return as a list of messages (the streaming pipeline expects an iterable)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


# --- ADD MEMORY TO THE GRAPH ---
memory = MemorySaver()
langchain_app = workflow.compile(checkpointer=memory)  # We'll call this "langchain_app" to avoid overshadowing FastAPI's 'app'


# --- FASTAPI ENDPOINT ---
@app.post("/stream/")
async def stream_handler(request_data: QueryRequest):
    """
    Streams responses from the compiled StateGraph app (langchain_app).
    Terminates streaming after some given seconds.
    """
    query = request_data.query
    thread_id = request_data.thread_id

    # Construct config for memory saver usage (thread tracking, etc.)
    config = {"configurable": {"thread_id": thread_id}}

    # Build the input messages for the model
    input_messages = [HumanMessage(query)]
    
    start_time = time.time()  # Start timer
    response_content = ""

    # Stream from the compiled graph
    for chunk, metadata in langchain_app.stream(
        {"messages": input_messages},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):
            response_content += chunk.content + " "

        # Stop if 300 seconds have passed
        if time.time() - start_time > 300:
            response_content += "\nStreaming ended after 300 seconds."
            break
    
    return {"response": response_content.strip()}


# --- LAUNCH SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)