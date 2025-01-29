import time
import hashlib
from typing import List, Dict
import numpy as np
import chromadb, os
from chromadb import Chroma
from chromadb.config import Settings

# Chroma의 임베딩 함수 설정 (예: OpenAIEmbeddings 사용)
# 실제 환경에 맞게 임베딩 함수 정의 필요
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

openai_key = os.getenv("OPENAI_API_KEY")
# 예시: OpenAI Embeddings 설정 (API 키 필요)
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_key,  # 실제 OpenAI API 키로 대체
    model_name="text-embedding-ada-002"  # 사용하고자 하는 임베딩 모델
)

class VectorDBManager:
    def __init__(self, collection_name='old_codes', persist_directory='chroma_db'):
        """
        Chroma 클라이언트와 컬렉션을 초기화합니다.
        
        :param collection_name: 사용할 컬렉션 이름
        :param persist_directory: Chroma DB의 퍼시스트 디렉토리
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Chroma 클라이언트 초기화
        self.client = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding_function
        )
        
        # 컬렉션 가져오기 또는 생성
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        두 벡터 간 코사인 유사도를 계산합니다.
        
        :param vec1: 첫 번째 벡터
        :param vec2: 두 번째 벡터
        :return: 코사인 유사도
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def generate_doc_id(self) -> str:
        """
        타임스탬프 기반의 고유한 doc_id를 생성합니다.
        
        :return: 생성된 doc_id 문자열
        """
        timestamp = int(time.time() * 1000)  # 밀리초 단위 타임스탬프
        return f"doc_{timestamp}"
    
    def save_to_vectordb(self, codes: List[Dict[str, str]]):
        """
        코드를 Chroma DB에 저장하기 전에 중복 여부를 검사하고, 
        필요 시 새로운 doc_id를 부여하여 저장합니다.
        
        :param codes: {'name': str, 'text': str} 형태의 코드 리스트
        """
        for code in codes:
            source_name = code.get("name")
            code_text = code.get("text")
            
            if not source_name or not code_text:
                print(f"Skipping incomplete code entry: {code}")
                continue
            
            # 새로운 코드의 임베딩 생성
            new_embedding = embedding_function(code_text)
            
            # 동일한 source 이름을 가진 기존 문서 검색
            existing_docs = self.collection.get(
                where={"source": source_name},
                include=["embeddings", "metadatas"]
            )
            
            existing_embeddings = existing_docs.get('embeddings', [])
            existing_metadatas = existing_docs.get('metadatas', [])
            
            duplicate_found = False
            for emb, meta in zip(existing_embeddings, existing_metadatas):
                similarity = self.cosine_similarity(new_embedding, emb)
                if similarity == 1.0:
                    # 완전히 동일한 코드 발견
                    duplicate_found = True
                    print(f"Duplicate code found for source '{source_name}'. Skipping save.")
                    break
            
            if not duplicate_found:
                # 새로운 doc_id 생성
                doc_id = self.generate_doc_id()
                
                # 메타데이터 설정
                doc_metadata = {
                    "source": source_name,
                    "doc_id": doc_id,
                    "timestamp": int(time.time())  # 초 단위 타임스탬프
                }
                
                # Chroma DB에 문서 추가
                self.collection.add(
                    documents=[code_text],
                    metadatas=[doc_metadata],
                    ids=[doc_id]
                )
                print(f"Saved code for source '{source_name}' with doc_id '{doc_id}'.")
    
    def is_duplicate(self, source_name: str, code_text: str) -> bool:
        """
        주어진 소스 이름과 코드 텍스트가 기존에 저장된 문서와 동일한지 확인합니다.
        
        :param source_name: 소스 이름
        :param code_text: 코드 텍스트
        :return: 동일한 문서가 존재하면 True, 아니면 False
        """
        # 새로운 코드의 임베딩 생성
        new_embedding = embedding_function(code_text)
        
        # 동일한 source 이름을 가진 기존 문서 검색
        existing_docs = self.collection.get(
            where={"source": source_name},
            include=["embeddings", "metadatas"]
        )
        
        existing_embeddings = existing_docs.get('embeddings', [])
        
        for emb in existing_embeddings:
            similarity = self.cosine_similarity(new_embedding, emb)
            if similarity == 1.0:
                return True
        return False
