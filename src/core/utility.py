from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import torch, os
import re, requests
import time

def save_to_vectordb(codes: List[Dict[str, str]]):
  """
  코드 조각을 벡터 DB에 저장하되, 동일한 파일(source)에서 코사인 유사도가 1.0인 문서는 제외하고 저장.
  """
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

  # Chroma 벡터 스토어 로드
  embedding = OpenAIEmbeddings()
  vectorstore = Chroma(
    collection_name="code_collection",
    persist_directory="chroma_db",
    embedding_function=embedding
  )

  new_documents = []
  
  for code in codes:
    source_name = code["name"]
    chunks = text_splitter.split_text(code["text"])

    for idx, chunk in enumerate(chunks):
      # 현재 청크를 벡터DB에서 검색하여 동일한 source 내 유사도 1.0인 문서가 있는지 확인
      similar_docs = vectorstore.similarity_search_with_score(chunk, k=5)  # 유사한 문서 최대 5개 검색

      has_duplicate = any(1 - score == 1.0 for _, score in similar_docs)  # 유사도 1.0 여부 확인
      if has_duplicate:
        print(f"⚠️ 동일한 코드 청크가 이미 존재하여 저장 생략: {source_name} (chunk {idx})")
        continue  # 중복이면 저장하지 않음
      
      # 고유한 doc_id 생성
      doc_id = f"{source_name}_chunk_{idx}_{int(time.time() * 1000)}"
      doc_metadata = {
        "source": source_name,
        "chunk_id": idx,
        "doc_id": doc_id,
        "timestamp": int(time.time())
      }
      
      new_documents.append((chunk, doc_metadata))

  # 새로운 문서만 추가로 저장
  if new_documents:
    vectorstore.add_texts(
      texts=[doc for (doc, meta) in new_documents],
      metadatas=[meta for (doc, meta) in new_documents]
    )
    vectorstore.persist()
    print(f"✅ {len(new_documents)}개의 코드 청크가 벡터DB에 저장되었습니다.")
  else:
    print("⚠️ 새로운 코드 청크가 없어 저장을 건너뜁니다.")

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

def search_prev_codes_from_vectordb(codes: List[Dict[str, str]]) -> List[str]:
  """
  현재 코드 리스트와 유사한 이전 코드를 검색하여 재구성.
  1) 현재 코드와 코사인 유사도가 1보다 낮은 문서들을 필터링
  2) 가장 최신(timestamp가 높은) 코드를 선택
  3) 해당 코드의 source 이름을 이용하여 벡터DB에서 관련 코드 청크를 검색
  4) chunk_id 순서대로 정렬하여 최종적으로 하나의 코드로 재조합

  :param codes: {'name': str, 'text': str} 형태의 코드 리스트
  :return: 이전 코드가 재구성된 리스트
  """
  prev_codes = []
  
  embedding = OpenAIEmbeddings()
  vectorstore = Chroma(
    collection_name="code_collection",
    persist_directory="chroma_db",
    embedding_function=embedding
  )

  for code in codes:
    source_name = code["name"]
    code_text = code["text"]

    # 현재 코드의 임베딩 생성
    # query_embedding = embedding.embed_query(code_text)  # 🔥 `embed_query()` 사용

    # 벡터DB에서 유사한 이전 코드 검색 (유사도 높은 순으로 최대 5개 검색)
    similar_docs = vectorstore.similarity_search_with_score(code_text, k=5)

    # 코사인 유사도 1.0(완전 동일한 코드)보다 낮은 것만 필터링
    filtered_candidates = [
      (doc, score) for doc, score in similar_docs if 1 - score < 1.0
    ]

    if not filtered_candidates:
      continue

    # timestamp가 가장 높은 후보 선택
    latest_candidate = max(filtered_candidates, key=lambda x: x[0].metadata["timestamp"])
    latest_source_name = latest_candidate[0].metadata["source"]

    # 동일한 source 이름을 가진 모든 청크 검색
    related_docs = vectorstore.get(
      where={"source": latest_source_name},
      include=["documents", "metadatas"]
    )

    # 청크를 chunk_id 순으로 정렬
    chunks_with_metadata = list(zip(related_docs["documents"], related_docs["metadatas"]))
    sorted_chunks = sorted(chunks_with_metadata, key=lambda x: x[1]["chunk_id"])

    # 청크 내용을 순서대로 결합
    reconstructed_code = "\n".join([chunk for (chunk, meta) in sorted_chunks])

    prev_codes.append(reconstructed_code)

  return prev_codes