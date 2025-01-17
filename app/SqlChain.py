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
    print(f"Question is: {question}")
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


def callSqlApi(info):
    sql_result = info["sql_result"]
    api_result = call_external_api(sql_result)

    formatted_result = {
        "data": api_result.get("data", "Fallback data"),  # API 결과의 'data'를 매핑
        "question": info.get("question", "Default question"),  # 원래 질문을 포함
    }
    return formatted_result


def get_sql_chain():

    firstChain = get_first_chain()

    # 두 번째 체인 생성
    second_chain = (
        {
            "sql_result": firstChain,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(callSqlApi)
        | secondPrompt
        | llm
        | StrOutputParser()
    )
    return second_chain
