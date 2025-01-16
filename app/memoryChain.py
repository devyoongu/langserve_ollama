from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 체인 생성
def create_memory_chain():

    retriever = st.session_state["retriever"]

    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.",
            ),
            # 대화기록용 key 인 chat_history 는 가급적 변경 없이 사용하세요!
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history
