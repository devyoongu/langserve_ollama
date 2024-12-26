import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

# 기록 파일 경로
EMBEDDINGS_RECORD_FILE = "embedded_files.txt"
VECTORSTORE_DIR = "vectorstores"  # 벡터스토어 저장 디렉토리


def is_file_embedded(file_path):
    """이미 임베딩된 파일인지 확인"""
    if not os.path.exists(EMBEDDINGS_RECORD_FILE):
        return False

    with open(EMBEDDINGS_RECORD_FILE, "r") as f:
        embedded_files = f.read().splitlines()
    return file_path in embedded_files


def record_embedded_file(file_path):
    """임베딩된 파일 기록"""
    with open(EMBEDDINGS_RECORD_FILE, "a") as f:
        f.write(file_path + "\n")


def get_vectorstore_path(file_path):
    """벡터스토어 파일 경로 반환"""
    file_name = os.path.basename(file_path)
    return os.path.join(VECTORSTORE_DIR, f"{file_name}.faiss")


def load_existing_retriever(file_path):
    """기존 벡터스토어에서 retriever 로드"""
    vectorstore_path = get_vectorstore_path(file_path)
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            vectorstore_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )
        print(f"[INFO] Loaded existing vectorstore for file '{file_path}'.")
        return vectorstore.as_retriever()
    else:
        print(f"[WARNING] Vectorstore for file '{file_path}' not found.")
        return None


def create_retriever(file_path=None):
    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    """파일을 임베딩하고 retriever를 생성합니다."""
    if file_path is None:
        # 모든 임베딩된 파일 확인
        if not os.path.exists(EMBEDDINGS_RECORD_FILE):
            warning_msg.error("파일을 업로드 해주세요.")
            return None

        with open(EMBEDDINGS_RECORD_FILE, "r") as f:
            embedded_files = f.read().splitlines()

        if embedded_files:
            # 첫 번째 임베딩된 파일의 vectorstore 로드
            first_embedded_file = embedded_files[0]
            return load_existing_retriever(first_embedded_file)
        else:
            warning_msg.error("파일을 업로드 해주세요.")
            return None

    # 이미 임베딩된 파일인지 확인
    if is_file_embedded(file_path):
        return load_existing_retriever(file_path)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 벡터스토어 저장
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore_path = get_vectorstore_path(file_path)
    vectorstore.save_local(vectorstore_path)

    # 단계 5: 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever()

    # 파일 기록
    record_embedded_file(file_path)

    print(f"[INFO] File '{file_path}' has been successfully embedded and recorded.")
    return retriever
