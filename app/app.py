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
import os
from retriever import create_retriever
from sidebar import render_sidebar
from chain import create_first_chain
from button import render_buttons
from initialize import initialize_environment, initialize_session

# API KEY 정보로드
load_dotenv()

st.title("RAG-YG-Action 프로젝트")

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] theDream RAG")

initialize_environment()
initialize_session()
selected_category = render_buttons()

# 사이드바 렌더링
uploaded_file, selected_model = render_sidebar()


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
# 파일 업로드 시 retriever 에서 중복 체크할 예정으로 cache 제거
# @st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    retriever = embed_file(uploaded_file)
    chain = create_first_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain
else:
    retriever = create_retriever()
    chain = create_first_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")


def process_input(input_text, chain):

    # 경고 메시지를 띄우기 위한 빈 영역
    warning_msg = st.empty()

    if chain is not None:
        # 사용자의 입력 처리
        st.chat_message("user").write(input_text)
        # 스트리밍 호출
        response = chain.stream(input_text)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록 저장
        add_message("user", input_text)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")


# 사용자 입력 처리
if user_input:
    process_input(user_input, st.session_state.get("chain"))

# 버튼 선택 처리
if selected_category:
    process_input(selected_category, st.session_state.get("chain"))
