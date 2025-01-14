import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

urlSet = ("https://spartacodingclub.kr/blog/all-in-challenge_winner",)
loader = WebBaseLoader(
    web_paths=urlSet,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("editedContent",)
        ),
        # features="html.parser",
        from_encoding="utf-8"  # 인코딩을 명시적으로 지정
    ),
)
docs = loader.load()
for idx, doc in enumerate(docs):
  print(f"idx; {idx + 1}, content: {doc.page_content}")
  
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(api_key=openai_key)
)

retriever = vectorstore.as_retriever()

user_msg = "ALL-in 코딩 공모전 수상작들을 요약해줘."
# retrieved_docs = retriever.invoke(user_msg)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
prompt = hub.pull("rlm/rag-prompt")

# user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_msg})
# print(user_prompt)

# response = llm.invoke(user_prompt)
# print(response.content)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke(user_msg))