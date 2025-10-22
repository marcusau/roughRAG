import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_xai import ChatXAI


from langchain.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()
os.environ["XAI_API_KEY"] = os.environ.get("XAI_API_KEY")
# os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY')

# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     max_tokens=30,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

# messages = [SystemMessage("You are a helpful assistant."),
#             HumanMessage("Hello, how are you?")]

# response = llm.invoke(messages)

# resp_content = response.content
# resp_metadata = response.response_metadata
# resp_additional_kwargs =  response.additional_kwargs
# resp_tool_calls = response.tool_calls
# resp_usage_metadata = response.usage_metadata

# total_tokens = resp_metadata['token_usage']
# prompt_tokens = total_tokens['prompt_tokens']
# completion_tokens = total_tokens['completion_tokens']

# print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
# print(f"Additional kwargs: {resp_additional_kwargs}")
# print(f"Tool calls: {resp_tool_calls}")
# print(f"Usage metadata: {resp_usage_metadata}")


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

llm = ChatXAI(
    model="grok-3-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)

structured_llm = llm.with_structured_output(Joke)
response = structured_llm.invoke([SystemMessage(content="You are a helpful assistant that tells jokes."),HumanMessage(content="Tell me a joke.")])
print(response)