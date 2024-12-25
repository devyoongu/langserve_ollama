from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

# 벡터 DB 저장 경로
FAISS_DB_PATH = "./faiss_db"


def create_retriever(file_path):
    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create or Load DB)
    if os.path.exists(FAISS_DB_PATH):
        # 기존 FAISS DB 로드 (위험한 직렬화 허용)
        vectorstore = FAISS.load_local(
            FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("INFO: Existing FAISS DB loaded from disk.")
    else:
        # 새롭게 FAISS DB 생성
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        # 디스크에 저장
        vectorstore.save_local(FAISS_DB_PATH)
        print("INFO: New FAISS DB created and saved to disk.")

    # 단계 5: 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever()
    return retriever
