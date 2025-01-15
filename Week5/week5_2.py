import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os, requests, time

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

url = "https://github.com/kasier48/DeepLearning/blob/main/Week5/test.py"
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
driver.get(url)

textarea_id = 'read-only-cursor-text-area'
# 페이지 로딩 대기 (필요 시 조정)
# 명시적 대기를 사용하여 특정 요소가 로드될 때까지 대기
try:
    element_present = EC.presence_of_element_located((By.ID, textarea_id))
    WebDriverWait(driver, 30).until(element_present)
except TimeoutException:
    print("페이지 로딩 시간 초과")

# 페이지 소스 가져오기
html = driver.page_source
driver.quit()

# BeautifulSoup으로 파싱
soup = bs4.BeautifulSoup(html, 'html.parser')
content_elements = soup.find_all(id=textarea_id)
print(f"찾은 요소 개수: {len(content_elements)}")

# 문서 객체 생성
docs = [Document(page_content=element.get_text(separator='\n', strip=True)) for element in content_elements]
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

user_msg = "코드를 리뷰하고 문제점을 말해줘."
# retrieved_docs = retriever.invoke(user_msg)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
prompt = hub.pull("rlm/rag-prompt")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke(user_msg))