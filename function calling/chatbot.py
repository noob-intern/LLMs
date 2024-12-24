import sys
import os
import getpass
import json
from typing import Optional
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver


os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter LangChain API Key: ")
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


"""
    Mock function to get guitar info from a database.
"""
def get_guitar_info(guitar_name: str) -> dict:
    mock_db = {
        "Acoustica Deluxe": {
            "body_material": "Solid Sitka Spruce",
            "neck_material": "Mahogany",
            "pickup_series": "ToneMaster Pickup Series",
            "price": "$2,500",
        },
        "Electric Pro X": {
            "body_material": "Alder",
            "neck_material": "Maple",
            "pickup_series": "ToneMaster Bridge Humbucker",
            "price": "$1,999",
        },
    }
    if guitar_name not in mock_db:
        return {
            "error": f"Guitar '{guitar_name}' not found in database."
        }
    return {
        "guitar_name": guitar_name,
        "details": mock_db[guitar_name]
    }


"""
    This schema describes a function that retrieves information about a guitar model, so GPT can call it
"""
openai_function_schemas = [
    {
        "name": "get_guitar_info",
        "description": "Get specs about a specific guitar from MelodyCraft Guitars' database",
        "parameters": {
            "type": "object",
            "properties": {
                "guitar_name": {
                    "type": "string",
                    "description": "The name of the guitar model to look up",
                },
            },
            "required": ["guitar_name"],
        },
    }
]


model = ChatOpenAI(
    model="gpt-4-0613",
    functions=openai_function_schemas,
    function_call="auto",
    temperature=0
)


SYSTEM_MESSAGE = """You are a friendly and knowledgeable customer service representative for Acoustica Guitars, 
a premium guitar manufacturer. You have access to the entire conversation so far as context.

IMPORTANT GUIDELINES:
- Do NOT repeat or copy-paste the entire previous conversation in your final output.
- Do NOT repeat boilerplate text or disclaimers each time. 
- Simply answer the user's latest question or request in a concise manner.
- You may reference context from earlier in the conversation, but do so succinctly.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


"""
    This function trims messages if they exceed a certain token limit. Currently, it is bypassed.
"""
def trimmer_invoke(messages):
    return messages



def call_model(state: MessagesState):
    """
        1) Trims messages if desired
        2) Builds the prompt
        3) Calls the model
        4) If the model returns a function call, intercept & resolve
        5) Optionally re-inject the function result into the conversation
    """

    trimmed_messages = trimmer_invoke(state["messages"])
    
    prompt = prompt_template.invoke({"messages": trimmed_messages})  # 2
    
    response = model.invoke(prompt)  # This returns an AIMessage
    

    function_call = response.additional_kwargs.get("function_call")
    if function_call:

        func_name = function_call.get("name")
        arguments_json = function_call.get("arguments")
        
        if func_name == "get_guitar_info":
            args_dict = json.loads(arguments_json)
            function_result = get_guitar_info(**args_dict)
            result_str = json.dumps(function_result, indent=2)
            
            # Option A: Return the function result as the final output
            # return {"messages": [AIMessage(content=result_str)]}
            
            # Option B: Re-inject the function result as a "function" message, 
            # then call the model again to let GPT produce a final user-facing answer.
            new_messages = trimmed_messages + [
                AIMessage(
                    role="function",
                    name=func_name,
                    content=result_str
                )
            ]
            second_prompt = prompt_template.invoke({"messages": new_messages})
            final_response = model.invoke(second_prompt)
            
            return {"messages": [final_response]}
    
    return {"messages": [response]}



"""
    State Graph
"""
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
langchain_app = workflow.compile(checkpointer=memory)



if __name__ == "__main__":
    conversation = []
    config = {
        "configurable": {
            "thread_id": "my-thread-abc123",
        }
    }

    print("Welcome to the Acoustica Guitars Chatbot! Type 'exit' or Ctrl+C to quit.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Add the user's message to the conversation
            conversation.append(HumanMessage(content=user_input))

            # Call the stateful app with the entire conversation
            result = langchain_app.invoke({"messages": conversation}, config=config)

            # The result is a dict with {"messages": [...]}, each item is either AIMessage or HumanMessage
            ai_responses = []
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage):
                    ai_responses.append(msg.content)
                    # IMPORTANT: append the AIMessage to conversation for the next turn
                    conversation.append(msg)

            # Print the final AI response(s) for the user
            print(f"AI: {ai_responses[-1]}")
            # for resp in ai_responses:
            #   print(f"AI: {resp}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            sys.exit(0)
