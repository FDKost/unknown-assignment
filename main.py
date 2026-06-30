from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent, AgentType
from langchain.schema import HumanMessage, AIMessage, ToolMessage

# Global list to store tool calls for printing
tool_calls = []

# Instantiate the local LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key="fake-api-key",
    base_url="http://localhost:1234/v1"
)

@tool
def get_price(product: str, city: str) -> str:
    """
    Generate a realistic price table for the product in the city.
    Returns a Markdown table with columns Product, Price (rub), Store.
    """
    # Record the tool call
    tool_calls.append(("get_price", {"product": product, "city": city}))

    # Create a sub-agent that will generate the price table
    sub_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        api_key="fake-api-key",
        base_url="http://localhost:1234/v1"
    )
    sub_agent = create_agent(
        llm=sub_llm,
        tools=[],
        agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        system_message="You are a price generator. Generate a realistic price table for the given product in the city."
    )

    prompt = f"Generate a realistic price table for {product} in {city}."
    result = sub_agent.invoke(prompt)

    # Extract the content from the AIMessage
    content = ""
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            content += msg.content

    return content.strip()

# Create the main agent with the get_price tool
main_agent = create_agent(
    llm=llm,
    tools=[get_price],
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    system_message="Ты помощник по планированию покупок"
)

# Sample query
query = "Помоги составить список покупок: молоко, хлеб, яблоки. Я нахожусь в Казани."

# Invoke the agent
answer = main_agent.invoke(query)

# Print all messages
print("\n--- Agent Conversation ---\n")
for idx, msg in enumerate(answer["messages"]):
    if isinstance(msg, HumanMessage):
        print(f"Human: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"Assistant: {msg.content}")
    elif isinstance(msg, ToolMessage):
        # Retrieve the corresponding tool call arguments
        if idx < len(tool_calls):
            tool_name, args = tool_calls[idx]
            print(f"Tool call: {tool_name} with arguments {args}")
        print(f"Tool result: {msg.content}")
    else:
        print(f"Unknown message type: {msg}")
