import os
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
from langchain_teddynote import logging

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] theDream RAG")

st.title("RAG-YG-Action 프로젝트")


def initialize_environment():
    """
    초기 환경 설정 및 디렉토리 생성
    """
    # 캐시 디렉토리 생성
    if not os.path.exists(".cache"):
        os.mkdir(".cache")

    # 파일 업로드 전용 폴더
    if not os.path.exists(".cache/files"):
        os.mkdir(".cache/files")

    # 임베딩 저장 폴더
    if not os.path.exists(".cache/embeddings"):
        os.mkdir(".cache/embeddings")

    if "chain" not in st.session_state:
        # 아무런 파일을 업로드 하지 않을 경우
        st.session_state["chain"] = None
        st.session_state["contact_data"] = {"name": "", "phone": "", "submitted": False}

    if "store" not in st.session_state:
        st.session_state["store"] = {}


def initialize_session():
    """
    Streamlit 세션 상태 초기화
    """
    if "messages" not in st.session_state:
        # 대화기록을 저장하기 위한 용도로 생성
        st.session_state["messages"] = []

        # 기본 인사말 추가
        st.session_state["messages"].append(
            ChatMessage(
                role="assistant",
                content="안녕하세요! 궁금한 내용을 물어보시면 도움을 드리겠습니다. 😊 아래 질문 유형 중 하나를 선택해 주세요.",
            )
        )
