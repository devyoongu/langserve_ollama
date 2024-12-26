from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# 벡터 DB 저장 경로
FAISS_DB_PATH = "./faiss_db"


def create_retriever(file_path=None):
    """
    Retriever 생성 함수. file_path가 None이면 기존 FAISS DB를 사용하고,
    그렇지 않으면 새 데이터를 처리하여 retriever를 반환.

    Args:
        file_path (str or None): 처리할 파일 경로. None이면 기존 DB 사용.

    Returns:
        retriever: 검색기 객체
    """
    # 단계 1: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # file_path가 None인 경우 기존 DB 로드
    if file_path is None:
        if os.path.exists(FAISS_DB_PATH):
            # 기존 FAISS DB 로드
            vectorstore = FAISS.load_local(
                FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            print("INFO: Existing FAISS DB loaded from disk.")
        else:
            raise FileNotFoundError(
                f"No existing FAISS DB found at {FAISS_DB_PATH}. Please provide a file_path to create a new DB."
            )
    else:
        # 단계 2: 문서 로드(Load Documents)
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

        # 단계 3: 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50
        )
        split_documents = text_splitter.split_documents(docs)

        # 단계 4: 새 DB 생성(Create DB)
        if os.path.exists(FAISS_DB_PATH):
            # 기존 DB 로드
            vectorstore = FAISS.load_local(
                FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            print("INFO: Existing FAISS DB loaded from disk and will be updated.")
        else:
            # 새 DB 생성
            vectorstore = FAISS.from_documents(
                documents=split_documents, embedding=embeddings
            )
            print("INFO: New FAISS DB created.")

        # 새 데이터를 추가하고 DB를 저장
        vectorstore.save_local(FAISS_DB_PATH)
        print("INFO: FAISS DB updated and saved to disk.")

    # 단계 5: 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever()
    return retriever
