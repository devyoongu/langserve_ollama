import streamlit as st
import requests
import threading
from langchain_core.messages.chat import ChatMessage
from retriever import default_retriever
from sidebar import render_sidebar
from button import render_buttons
from initialize import initialize_environment, initialize_session
import time
from file_watch import (
    start_file_watch,
)  # file_watch.py에서 start_file_watch 함수 가져오기


# 폴더 감시 스레드 시작 (Streamlit 로드 시 자동 실행)
def initialize_file_watch():
    def run_file_watch():
        start_file_watch()

    watch_thread = threading.Thread(target=run_file_watch, daemon=True)
    watch_thread.start()
    st.session_state["file_watch_initialized"] = True


# 폴더 감시 초기화가 되어 있지 않으면 시작
if "file_watch_initialized" not in st.session_state:
    initialize_file_watch()
    st.info("폴더 감시가 시작되었습니다. 새로운 파일을 업로드하세요.")

initialize_environment()
initialize_session()
render_sidebar()
default_retriever()
selected_category = render_buttons()


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 대화이력 전송 API
def send_chat_log_to_api(chat_logs):
    url = "http://localhost:8080/api/chat-log"
    payload = {
        "chatThreadId": st.session_state.get("chat_thread_id"),
        "chatLogs": chat_logs,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    if "data" in response_data and "id" in response_data["data"]:
        st.session_state["chat_thread_id"] = response_data["data"]["id"]


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 사용자 입력 처리 함수
def process_input(input_text, chain):
    warning_msg = st.empty()

    if chain is not None:
        # 사용자 메시지 출력
        st.chat_message("user").write(input_text)
        user_message_time = int(time.time())

        # 스트리밍 호출
        response = chain.stream(input_text)
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
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
    else:
        warning_msg.error("파일을 업로드 해주세요.")


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")


# 사용자 입력 처리
if user_input:
    process_input(user_input, st.session_state.get("chain"))

# 버튼 선택 처리
if selected_category:
    process_input(selected_category, st.session_state.get("chain"))
