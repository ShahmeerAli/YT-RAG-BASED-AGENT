import os
from dot_env import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    huggingfacehub_api_token=api_token,

)
model=ChatHuggingFace(llm=llm)