import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from faiss import IndexFlatL2
from fastapi import UploadFile
from chainUtils import ALLOWED_TYPES

# 기록 파일 경로
EMBEDDINGS_RECORD_FILE = "embedded_files.txt"
EMBEDDING_DIM = 1536  # OpenAI 기본 임베딩 차원


def is_file_embedded(file_path):
    """이미 임베딩된 파일인지 확인"""
    if not os.path.exists(EMBEDDINGS_RECORD_FILE):
        return False

    with open(EMBEDDINGS_RECORD_FILE, "r") as f:
        embedded_files = f.read().splitlines()
    return file_path in embedded_files


def record_embedded_file(file_path):
    """
    임베딩된 파일과 관련 메타 ID를 기록합니다.
    """
    with open(EMBEDDINGS_RECORD_FILE, "a") as f:
        f.write(f"{file_path}\n")  # 파일 경로를 CSV 형식으로 저장


def load_existing_retriever(type):
    """기존 벡터스토어에서 retriever 로드"""
    # 타입 유효성 검사
    if type not in ALLOWED_TYPES:
        raise ValueError(
            f"Invalid type '{type}'. Allowed types are: {', '.join(ALLOWED_TYPES)}"
        )

    # 타입별 벡터스토어 경로 설정
    vectorstore_path = f"vectorstores/{type}_vectorstore.faiss"
    embeddings = OpenAIEmbeddings()

    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )
        print(
            f"[INFO] Loaded existing vectorstore for type '{type}' from '{vectorstore_path}'."
        )
    else:
        print(
            f"[WARNING] Vectorstore for type '{type}' at '{vectorstore_path}' not found. Returning empty retriever."
        )
        # 빈 검색기 생성
        index = IndexFlatL2(EMBEDDING_DIM)
        docstore = {}
        vectorstore = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id={},
            embedding_function=embeddings,
        )

    # type에 따라 session_state에 retriever 설정
    retriever_key = f"{type}_retriever"
    if type == "document":
        st.session_state[retriever_key] = vectorstore.as_retriever()
    elif type == "department":
        st.session_state[retriever_key] = vectorstore.as_retriever()

    return vectorstore.as_retriever()


def create_retriever(file_path, type):
    # 이미 임베딩된 파일인지 확인
    if is_file_embedded(file_path):
        return load_existing_retriever(type)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: 기존 벡터스토어 로드 또는 새로 생성
    vectorstore_path = (
        f"vectorstores/{type}_vectorstore.faiss"  # 타입별로 다른 경로 설정
    )
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(split_documents)
    else:
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )

    # 벡터스토어 저장
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    vectorstore.save_local(vectorstore_path)

    # 파일 기록
    record_embedded_file(file_path)

    print(
        f"[INFO] File '{file_path}' with type '{type}' has been successfully embedded and added to the vectorstore."
    )
    retriever_key = f"{type}_retriever"
    st.session_state[retriever_key] = vectorstore.as_retriever()
    return vectorstore.as_retriever()


def process_file(uploaded_file, type):
    # 타입 유효성 검사
    print(f"upload type is {type}")
    if type not in ALLOWED_TYPES:
        raise ValueError(
            f"Invalid type '{type}'. Allowed types are: {', '.join(ALLOWED_TYPES)}"
        )
    print(f"[INFO] uploaded_file is '{uploaded_file}'.")
    file_path = save_file(uploaded_file)
    create_retriever(file_path, type)


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
def save_file(file):
    file_content = file.read()
    file_path = f"./upload/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path


async def process_file_fastapi(uploaded_file):
    """
    FastAPI용 파일 처리 메서드
    """
    print(f"[INFO] uploaded_file is '{uploaded_file.filename}'.")
    file_path = await save_file_fastapi(uploaded_file)
    create_retriever(file_path)
    print("[INFO] Chain created successfully.")


async def save_file_fastapi(file: UploadFile):
    """
    FastAPI용 파일 저장 메서드
    """
    file_content = await file.read()  # 비동기 파일 읽기
    file_path = f"./upload/{file.filename}"  # file.filename 사용
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 디렉토리 생성
    with open(file_path, "wb") as f:
        f.write(file_content)
    print(f"[INFO] File saved at '{file_path}'.")
    return file_path
