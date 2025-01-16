from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from chainUtils import get_session_history


# 체인 생성
def create_general_chain():

    # 프롬프트 정의
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.  \
        Use the following pieces of retrieved context to answer the question. \
        Check first Previous Chat History. 
        Answer in Korean.

        #Previous Chat History:
        {chat_history}

        Question: {question}
        Answer:"""
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 일반 Chain 생성
    # chain = prompt | llm | StrOutputParser()
    # 단계 8: 체인(Chain) 생성
    chain = (
        {
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_history
