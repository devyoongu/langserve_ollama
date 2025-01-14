import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from faiss import IndexFlatL2
from ragChain import get_rag_chain
from fastapi import UploadFile

# 기록 파일 경로
EMBEDDINGS_RECORD_FILE = "embedded_files.txt"
VECTORSTORE_PATH = "vectorstores/main_vectorstore.faiss"  # 통합 벡터스토어 경로
EMBEDDING_DIM = 1536  # OpenAI 기본 임베딩 차원


def is_file_embedded(file_path):
    """이미 임베딩된 파일인지 확인"""
    if not os.path.exists(EMBEDDINGS_RECORD_FILE):
        return False

    with open(EMBEDDINGS_RECORD_FILE, "r") as f:
        embedded_files = f.read().splitlines()
    return file_path in embedded_files


def record_embedded_file(file_path, vector_meta_id):
    """
    임베딩된 파일과 관련 메타 ID를 기록합니다.
    """
    with open(EMBEDDINGS_RECORD_FILE, "a") as f:
        f.write(
            f"{file_path},{vector_meta_id}\n"
        )  # 파일 경로와 메타 ID를 CSV 형식으로 저장


def load_existing_retriever():
    """기존 벡터스토어에서 retriever 로드"""
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print(f"[INFO] Loaded existing vectorstore from '{VECTORSTORE_PATH}'.")
        return vectorstore.as_retriever()
    else:
        print(
            f"[WARNING] Vectorstore '{VECTORSTORE_PATH}' not found. Returning empty retriever."
        )
        # 빈 검색기 생성
        index = IndexFlatL2(EMBEDDING_DIM)
        docstore = {}
        return FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id={},
            embedding_function=embeddings,
        ).as_retriever()


def default_retriever():
    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    # 통합 벡터스토어 로드
    retriever = load_existing_retriever()
    if retriever is None:
        warning_msg.error("Vectorstore가 존재하지 않습니다. 파일을 업로드 해주세요.")
    return retriever


def create_retriever(file_path):
    # 이미 임베딩된 파일인지 확인
    if is_file_embedded(file_path):
        return load_existing_retriever()

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: 기존 벡터스토어 로드 또는 새로 생성
    vector_meta_id = generate_vector_meta_id(file_path)
    for doc in split_documents:
        doc.metadata["vector_meta_id"] = vector_meta_id  # 메타 ID 추가

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(split_documents)
    else:
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )

    # 벡터스토어 저장
    os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    # 파일 기록
    record_embedded_file(file_path, vector_meta_id)

    print(
        f"[INFO] File '{file_path}' has been successfully embedded and added to the vectorstore."
    )
    return vectorstore.as_retriever()


def generate_vector_meta_id(file_path):
    """
    벡터스토어에서 메타 ID를 생성합니다.
    """
    # 간단히 해시값으로 ID 생성
    import hashlib

    vector_meta_id = hashlib.sha256(
        f"{file_path}{VECTORSTORE_PATH}".encode()
    ).hexdigest()
    print(f"[INFO] Generated vector meta ID: {vector_meta_id}")
    return vector_meta_id


def process_file(uploaded_file):
    print(f"[INFO] uploaded_file is '{uploaded_file}'.")
    file_path = save_file(uploaded_file)
    retriever = create_retriever(file_path)
    st.session_state["retriever"] = retriever
    chain = get_rag_chain()
    st.session_state["chain"] = chain


def process_without_file():
    retriever = default_retriever()
    st.session_state["retriever"] = retriever
    chain = get_rag_chain()
    st.session_state["chain"] = chain


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
