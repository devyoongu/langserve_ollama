from pyexpat import model
import streamlit as st
import requests
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from dotenv import load_dotenv
from retriever import process_file, process_without_file
from memoryRoutingChain import get_router_chain
from sidebar import render_sidebar
from button import render_buttons
from initialize import initialize_environment, initialize_session
import time
from llmApi import send_chat_log_to_api


# API KEY 정보로드
load_dotenv()

st.title("RAG-YG-Action 프로젝트")

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] theDream RAG")

initialize_environment()
initialize_session()
selected_category = render_buttons()

# 사이드바 렌더링
uploaded_file = render_sidebar()


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일이 업로드 되었을 때
if uploaded_file:
    process_file(uploaded_file)
else:
    process_without_file()
    get_router_chain()


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")


# 사용자 입력 처리 함수
def process_input(input_text):
    # warning_msg = st.empty()

    router_chain = st.session_state.get("router_chain")
    chat_thread_id = st.session_state.get("chat_thread_id")

    print(f"session chat_thread_id is {chat_thread_id}")

    # 사용자 메시지 출력
    st.chat_message("user").write(input_text)
    user_message_time = int(time.time())

    # invoke 호출로 응답 받기
    response = router_chain.stream(
        {"question": input_text},
        config={"configurable": {"session_id": chat_thread_id}},
    )

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""

        # 스트리밍 데이터 처리
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화 기록 저장
    add_message("user", input_text)
    add_message("assistant", ai_answer)

    # API 요청
    chat_logs = [
        {"role": "user", "content": input_text, "createdTime": user_message_time},
        {
            "role": "assistant",
            "content": ai_answer,
            "createdTime": int(time.time()),
        },
    ]
    send_chat_log_to_api(chat_logs)


def process_button(input_text):

    rag_chain = st.session_state.get("chain")

    # 사용자 메시지 출력
    st.chat_message("user").write(input_text)

    # 스트리밍 호출
    response = rag_chain.stream(input_text)
    with st.chat_message("assistant"):
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화 기록 저장
    add_message("user", input_text)
    add_message("assistant", ai_answer)


# 사용자 입력 처리
if user_input:
    process_input(user_input)

# 버튼 선택 처리
if selected_category:
    process_input(selected_category)
