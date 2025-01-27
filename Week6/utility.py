from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
import re, requests

def save_to_vectordb(codes):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

  documents = []
  for code in codes:
    chunks = text_splitter.split_text(code["text"])
    for idx, chunk in enumerate(chunks):
      doc_metadata = {"source": code["name"], "chunk_id": idx}
      documents.append((chunk, doc_metadata))

  embedding = OpenAIEmbeddings()
  vectorstore = Chroma.from_texts(
    texts=[doc for (doc, meta) in documents],
    embedding=embedding,
    metadatas=[meta for (doc, meta) in documents],
    collection_name='old_codes',
    persist_directory="chroma_db"
  )

def get_review_prompt(history, prev_codes, codes):
  prompt_template = """\
  이전 대화 내역 입니다.
  {history}
  
  아래는 이전 코드 조각들입니다:
  {context}

  그리고 아래는 새로 제출된 코드입니다:
  {new_code}

  위 정보를 참고하여 다음을 수행하세요:
  1. 새 코드가 이전 코드에 비해 어떤 부분이 변경되었는지 요약
  2. 개선해야 할 점, 잠재적 버그, 리팩토링 포인트 등 코드 리뷰 포인트 제시

  답변을 한국어로 해주세요.
  """

  context = '\n\n'.join(prev_codes)
  new_codes = '\n\n'.join([code['text'] for code in codes])
  
  template = PromptTemplate(
      template=prompt_template,
      input_variables=["history", "context", "new_code"]
  )
  final_prompt = template.format(history=history, context=context, new_code=new_codes)
  return final_prompt

def parse_codes(prompt):
  github_urls = []
  pattern = r'https://github\.com/(\S+)'
  matches = re.findall(pattern, prompt)
  if matches:
    for github_url in matches:
      github_url = 'https://raw.githubusercontent.com/' + github_url.replace('blob', 'refs/heads')
      
      if github_url not in github_urls:
        github_urls.append(github_url)
        
  codes = []
  for github_url in github_urls:
    split_text = github_url.split('/')
    name = split_text[-1]
    response = requests.get(github_url)
    codes.append(
      {
        'name': name,
        'text': response.text
      }
    )
  
  return codes

def search_prev_codes_from_vectordb(codes):
  prev_codes = []
  
  embedding = OpenAIEmbeddings()
  vectorstore = Chroma(
      collection_name="old_codes",
      persist_directory="chroma_db",
      embedding_function=embedding
  )
  
  for code in codes:
    text = code['text']
    similar_docs = vectorstore.similarity_search(text, k=1)
    
    for doc in similar_docs:
      prev_codes.append(doc.page_content)

  return prev_codes