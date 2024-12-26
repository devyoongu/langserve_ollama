from pyexpat import model
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
from retriever import create_retriever
import requests  # requests 모듈을 추가로 임포트합니다.


# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Local SQL DDL 문서 기반 RAG 채봇 서비스 💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["ollama", "xionic"], index=0)


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)


# retriever 를 통해 검색된 document_list는 메타정보까지 모두 포함되어 있기 때문에 format_doc을 체이닝으로 추가하여 content 내용만 연결된 String 값으로 리턴
# 불필요한 메터정보를 제외한 content 내용만 붙여서 yaml 템플릿의 context에 넣기 좋기 변환해주는 역할
def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성
def create_first_chain(retriever, model_name="ollama"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.

    print("INFO: Using 'ollama' model configuration.")

    # 단계 6: 프롬프트 생성(Create Prompt)
    prompt = load_prompt("prompts/00_sql-generator.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # Ollama 모델을 불러옵니다.
    llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def create_second_chain():
    # 2차 체인에 사용할 프롬프트 및 모델 설정
    prompt = load_prompt("prompts/01_table-generator.yaml", encoding="utf-8")
    llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)

    # 2차 체인 생성
    second_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "data": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return second_chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    retriever = embed_file(uploaded_file)
    # 업로드될 때 임베딩이 되고 바로 체인이 생성되는데 질문이 들어오면 이 체인을 session 에서 그대로 사용하기 때문에 프롬프트로 넘어가는 retriever 된 context가 전체가 넘어가는 듯하다 (-> 사용자 질문 관련만 조회되어야 한다. )
    chain = create_first_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()


# 만약에 사용자 입력이 들어오면...
# 사용자의 입력 처리
def process_first_chain(user_input):
    """1차 체인 생성 및 실행."""
    first_chain = st.session_state.get("chain")
    if not first_chain:
        warning_msg.error("파일을 업로드 해주세요.")
        return None

    # 사용자의 입력을 즉시 화면에 표시
    st.chat_message("user").write(user_input)

    # 체인을 호출하여 응답 생성
    ai_answer = first_chain.invoke(user_input)

    # 체인의 응답을 화면에 표시
    with st.chat_message("assistant"):
        st.markdown(ai_answer)

    return ai_answer


def call_external_api(ai_answer):
    """외부 API 호출."""
    api_url = "http://localhost:8080/api/sqldeck/execute"
    api_headers = {"Content-Type": "application/json"}
    api_body = {"sqlQuery": ai_answer}
    try:
        api_response = requests.post(api_url, headers=api_headers, json=api_body)
        api_response.raise_for_status()
        return api_response.json()
    except requests.RequestException as e:
        st.chat_message("assistant").write(f"External API call failed: {str(e)}")
        return {"error": "API call failed"}


def process_second_chain(user_input, api_result):
    """2차 체인 생성 및 실행."""
    second_chain = create_second_chain()
    second_response = second_chain.stream(
        {
            "context": "",
            "question": user_input,
            "data": api_result.get("data", "Fallback question"),
        }
    )
    with st.chat_message("assistant"):
        second_container = st.empty()
        second_ai_answer = ""
        for token in second_response:
            second_ai_answer += token
            second_container.markdown(second_ai_answer)
    return second_ai_answer


# 메인 로직
if user_input:
    # 사용자의 입력을 바로 화면에 표시하고 1차 체인을 처리
    ai_answer = process_first_chain(user_input)
    if not ai_answer:
        warning_msg.error("1차 체인 응답 처리 중 오류가 발생했습니다.")
        # return

    # 외부 API 호출
    api_result = call_external_api(ai_answer)

    # 2차 체인 처리
    second_ai_answer = process_second_chain(user_input, api_result)

    # 대화 기록 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)  # 1차 체인 답변
    add_message("assistant", second_ai_answer)  # 2차 체인 답변
