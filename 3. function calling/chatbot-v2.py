import sys
import os
import getpass
import json
from typing import Optional
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
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



GPT_MODEL = "gpt-4o"
client = OpenAI()

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    """
    Utility to call the Chat Completions API with exponential backoff retry.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    """
    Print conversation with colored role labels.
    """
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))



def call_model(state: MessagesState):
    """
    1) Trims messages if desired
    2) Builds the prompt
    3) Calls the model
    4) If the model returns a function call, intercept & resolve
    5) Optionally re-inject the function result into the conversation
    """
    trimmed_messages = state["messages"]
    
    # 2) Build the prompt
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    
    # 3) Call the model
    response = model.invoke(prompt)

    # 4) Check if the model returned a function call
    function_call = response.additional_kwargs.get("function_call")
    if function_call:
        func_name = function_call.get("name")
        arguments_json = function_call.get("arguments")
        
        if func_name == "get_guitar_info":
            args_dict = json.loads(arguments_json)
            function_result = get_guitar_info(**args_dict)
            result_str = json.dumps(function_result, indent=2)
            
            # Option B: Re-inject the function result as a "function" message
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
    
    # If no function call, just return the response
    return {"messages": [response]}



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

            # The result is a dict with {"messages": [...]}
            ai_responses = []
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage):
                    ai_responses.append(msg.content)
                    # IMPORTANT: append the AIMessage to conversation for the next turn
                    conversation.append(msg)

            # Print the final AI response(s) for the user
            print(f"AI: {ai_responses[-1]}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            sys.exit(0)