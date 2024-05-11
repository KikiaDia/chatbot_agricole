from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableParallel
from langchain_core.prompts import format_document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.base import DEFAULT_DOCUMENT_PROMPT
from glob import glob
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware


# function 
def format_docs(inputs: dict) -> str:
  inputs["context"] = "\n\n".join(
      f"Document {i}:\n {format_document(doc, DEFAULT_DOCUMENT_PROMPT)}" for i, doc in enumerate(inputs["context"])
  )
  return inputs

def format_chat_history(data):
  formatize_chat_history = ""
  if "chat_history" in data.keys():
    for message in data["chat_history"]:
      formatize_chat_history += f"\t{str(type(message)).split("'")[1].split(".")[-1]}: {message.content.replace("\n", "")}\n"
    data["chat_history"] = formatize_chat_history
  return data

def format_response(response: str) -> dict:
  return {"input": response}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ingestion
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

pages = []
for document in glob(pathname="app/documents/*"):
  loader = PyPDFLoader(document)
  pages.extend(loader.load_and_split(text_splitter=splitter))

# llm
llm = ChatOpenAI(
  openai_api_base="https://api.groq.com/openai/v1",
  model = "llama3-70b-8192",
  temperature=0.7,
  api_key="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"
)

# indexing
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", show_progress=True)
vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)

# retriever
retriever = vectorstore.as_retriever()

# QA Prompt
qa_system_prompt = """
Je suis un assistant agricole intelligent qui fournit des recommandations sur l'irrigation, la fertilisation, les traitements 
phytosanitaires et les meilleures pratiques agricoles. Sur la base du contexte, des informations de l'utilisateur et la question 
de l'utilisateur, fournit des recommandations à l'utilisateur de manière précise sur comment il pourrait agir pour améliorer son
rendement.

contexte : {context}
"""
#informations: {info}

qa_prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "Je suis Mbaykat🌾🦾🤖🧠 une solution de recommandations agricoles basée sur l'IA Générative."),
    ("system", qa_system_prompt),
    ("human", "{input}"),
  ]
)

# Contextualization
contextualize_query_system_prompt = """
Je suis un assistant IA très util pour contextualiser les fils de discusssion. Compte tenu de l’historique du chat et de la 
dernière question de l’utilisateur, formule une question autonome qui peut être compris sans l’historique du chat.
Donne juste la question directe et brute qui sera la proche possible de la dernière question de l'utilisateur.

NE REPONDEZ PAS A LA QUESTION, reformulez-la au besoin ou retournez-la tel quel sinon.

Historique_Chat: 
{chat_history}

Dernière_Question: {input}
"""

contextualize_query_prompt = ChatPromptTemplate.from_template(contextualize_query_system_prompt)

contextualization_chain = RunnableBranch(
  (
    lambda x: not x.get("chat_history", False) ,
    lambda x: x["input"],
  ),
  format_chat_history | contextualize_query_prompt | llm | StrOutputParser(),
)

retrieval_chain = {"context": retriever , "input": RunnablePassthrough()}

chain = contextualization_chain | retrieval_chain | qa_prompt | llm | StrOutputParser()

store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, conversational_rag_chain, path="/mbaykat")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)