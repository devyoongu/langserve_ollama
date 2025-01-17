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
from langchain_core.runnables import RunnableLambda
import requests
from langchain_teddynote.prompts import load_prompt
from llmApi import call_external_api


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


def fetch_context(question, retriever):
    # print(f"fetch_context question is {question}")
    documents = retriever.get_relevant_documents(question)
    return "\n\n".join(doc.page_content for doc in documents)


# llm 생성
llm = ChatOpenAI(model_name="gpt-4o-mini")

# 프롬프트 정의
firstPrompt = load_prompt("prompts/00_sql-generator.yaml", encoding="utf-8")
secondPrompt = load_prompt("prompts/01_table-generator.yaml", encoding="utf-8")


# 1차 체인 생성
def get_first_chain():

    retriever = st.session_state["retriever"]

    # 일반 Chain 생성
    chain = (
        {
            "context": lambda x: fetch_context(x["question"], retriever),
            "question": itemgetter("question"),
        }
        | firstPrompt
        | llm
        | StrOutputParser()
    )

    return chain


# 1차 체인 생성
def get_second_chain():
    # 2차 체인에 사용할 프롬프트 및 모델 설정

    # 2차 체인 생성
    second_chain = (
        {
            "question": RunnablePassthrough(),
            "data": RunnablePassthrough(),
        }
        | second_chain
        | llm
        | StrOutputParser()
    )
    return second_chain


def route(info):
    if not isinstance(info, dict):
        print(f"Invalid input for route: {info}")
        raise ValueError("Input to route function must be a dictionary.")

    api_result = call_external_api(info["sqlQuery"])

    second_chain = (
        {
            "question": RunnablePassthrough(),
            "data": api_result.get("data", "Fallback question"),
        }
        | secondPrompt
        | llm
        | StrOutputParser()
    )

    return second_chain


def get_sql_chain():
    second_chain = (
        {
            "sqlQuery": get_first_chain(),  # chain 실행 결과를 적절히 변환 lambda x: 추가?
            "question": itemgetter("question"),
        }
        | RunnableLambda(route)
        | StrOutputParser()
    )

    return second_chain
