import chromadb
import streamlit as st

from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@st.cache_resource
def get_docs(pdf_stream):
  with open('tmp.pdf', 'wb') as f:
      f.write(pdf_stream.getvalue())
  loader = PyPDFLoader("tmp.pdf")

  docs = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  return splits


def get_retriever(docs):
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    splits = get_docs(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever


rag_prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.title("RAG Bot")
model = ChatOpenAI(model="gpt-4o-mini")
if docs := st.file_uploader("Upload your PDF here and click", type="pdf"):
    retriever = get_retriever(docs)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("What is your question about the PDF?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            retrieved_docs = retriever.invoke(prompt)
            user_prompt = rag_prompt.invoke({"context": format_docs(retrieved_docs), "question": prompt})
            result = model.invoke(user_prompt)

            response = result.content
            st.markdown(response)

		        st.session_state.messages.append({
		            "role": "assistant",
		            "content": response
		        })