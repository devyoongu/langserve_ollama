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
from langchain_ollama import ChatOllama
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
from retriever import create_retriever

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

    # 기본 인사말 추가
    st.session_state["messages"].append(
        ChatMessage(
            role="assistant",
            content="안녕하세요! 궁금한 내용을 물어보시면 도움을 드리겠습니다. 😊 아래 질문 유형 중 하나를 선택해 주세요.",
        )
    )
# 버튼 생성 (질문 유형)
selected_category = None
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("위하고에 대해 알려줘"):
        selected_category = "위하고에 대해 알려줘"
with col2:
    if st.button("we톡에 대해 알려줘"):
        selected_category = "we톡에 대해 알려줘"
with col3:
    if st.button("서울의 수도는 어디야?"):
        selected_category = "서울의 수도는 어디야?"


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

    # 폼 UI
    st.write("상담이 필요한 경우 연락처를 남겨주세요")

    # 연락처 FORM 추가
    with st.form(key="contact_form"):
        # 지역 선택 추가
        region = (
            st.selectbox(
                "Select your region",
                [
                    "Seoul",
                    "Busan",
                    "Daegu",
                    "Incheon",
                    "Gwangju",
                    "Daejeon",
                    "Ulsan",
                    "Other",
                ],
            ),
        )
        name = st.text_input("Name")
        phone_number = st.text_input("Phone Number")
        submit_button = st.form_submit_button(label="Submit")

    # 버튼 클릭 후 상태 관리
    if submit_button:
        st.session_state["form_submitted"] = True

        # ChatMessage 객체를 직렬화 가능한 딕셔너리로 변환
        serialized_messages = [
            {"role": message.role, "content": message.content}
            for message in st.session_state["messages"]
        ]

        # 세션 상태에 저장
        st.session_state["form_data"] = {
            "region": region[0],
            "name": name,
            "phoneNumber": phone_number,
            "dialogues": serialized_messages,
        }

        st.write("Form submitted successfully!")
        st.write(st.session_state["form_data"])

    # API 요청 처리
    if st.session_state.get("form_submitted", False):
        form_data = st.session_state.get("form_data", {})
        try:
            response = requests.post(
                "http://localhost:8080/api/contact",
                json=form_data,
            )
            if response.status_code == 200:
                st.success("감사합니다! 연락처가 성공적으로 제출되었습니다.")
                st.session_state["form_submitted"] = False  # 상태 초기화
            else:
                st.error(f"요청이 실패했습니다. 상태 코드: {response.status_code}")
                st.error(f"응답 메시지: {response.text}")
        except Exception as e:
            st.error(f"API 요청 중 오류 발생: {str(e)}")


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
    prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")

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


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    retriever = embed_file(uploaded_file)
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
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")


# 선택된 버튼을 처리
if selected_category:
    # 선택된 카테고리를 질문으로 변환
    # category_question = f"{selected_category}에 대해 알려줘"

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(selected_category)
        # 스트리밍 호출
        response = chain.stream(selected_category)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", selected_category)
        add_message("assistant", ai_answer)

    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")


# 연락처 입력 상태 초기화
if "contact_data" not in st.session_state:
    st.session_state["contact_data"] = {"name": "", "phone": "", "submitted": False}


# 폼 초기화를 위한 함수
def reset_contact_form():
    st.session_state["contact_data"] = {"name": "", "phone": "", "submitted": False}


# API 요청 함수
def submit_contact_form():
    contact_data = st.session_state["contact_data"]
    payload = {
        "name": contact_data["name"],
        "phoneNumber": contact_data["phone"],
        "question": selected_category if selected_category else "질문 없음",
    }
    try:
        response = requests.post("http://localhost:8080/api/contact", json=payload)
        if response.status_code == 200:
            st.success("감사합니다! 연락처가 성공적으로 제출되었습니다.")
            contact_data["submitted"] = True
        else:
            st.error(f"요청이 실패했습니다. 상태 코드: {response.status_code}")
            st.error(f"응답 메시지: {response.text}")
    except Exception as e:
        st.error(f"API 요청 중 오류 발생: {str(e)}")


# API 호출 함수 (추가된 부분)
def save_session_data():
    if "form_data" in st.session_state:
        try:
            response = requests.post(
                "http://localhost:8080/api/api/contact",  # API 엔드포인트
                json=st.session_state["form_data"],  # 세션 상태 데이터를 전송
            )
            if response.status_code == 200:
                st.success("세션 데이터가 성공적으로 저장되었습니다.")
            else:
                st.error(f"세션 데이터 저장 실패: {response.status_code}")
        except Exception as e:
            st.error(f"API 호출 중 오류 발생: {str(e)}")


# 창 닫기 및 새로고침 시 이벤트 처리 (추가된 부분)
st.markdown(
    """
    <script>
        window.addEventListener("beforeunload", function (e) {
            // 이 이벤트가 발생할 때 서버에 저장 요청
            fetch("/save-session-data", {method: "POST"}); // Streamlit 서버 연결 필요
        });
    </script>
    """,
    unsafe_allow_html=True,
)

# Streamlit에서 매번 실행되도록 설정
if st.session_state.get("form_data"):
    save_session_data()
