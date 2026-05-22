import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    huggingfacehub_api_token=api_token,

)
model=ChatHuggingFace(llm=llm)

# fetching the video via video id

video_id="ldxFjLJ3rVY"
try:
    yttp=YouTubeTranscriptApi()
    transcript_list = yttp.fetch(video_id, languages=['en'])
    transcript=" ".join(chunk.text for chunk in transcript_list )
except TranscriptsDisabled:
    print("Transcript is not available for this video")

# now using text splitters for this

splitterrs=RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=0
)

texts=splitterrs.create_documents([transcript])

print(len(texts))


embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store=FAISS.from_documents(
    texts,embeddings
)


#retrieval step 

retrivers=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

print(retrivers.invoke("what does the endpoints of the line on the top represent?"))

