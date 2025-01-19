from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from memoryRagChain import create_rag_chain
from memoryGeneralChain import create_general_chain
from operator import itemgetter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from chainUtils import get_session_history
from SqlChain import get_sql_chain

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", stream=True)

prompt = load_prompt("prompts/00_route.yaml", encoding="utf-8")

general_prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.  \
    Use the following pieces of retrieved context to answer the question. \
    Check first Previous Chat History. 
    Answer in Korean.

    #Previous Chat History:
    {chat_history}

    Question: {question}
    Answer:"""
)


# 체인을 생성합니다.
chain = prompt | llm | StrOutputParser()  # 문자열 출력 파서를 사용합니다.


def route(info):
    # 입력 데이터 형식 검증
    if not isinstance(info, dict):
        print(f"Invalid input for route: {info}")
        raise ValueError("Input to route function must be a dictionary.")

    if "법무법인" in info["topic"].lower():
        print("Routing to rag_chain from session_state (법무법인 관련)")
        # return get_route_rag_chain()
        return create_rag_chain()
    elif "연락처" in info["topic"].lower():
        print("Routing to 연락처 chain")
        # return general_chain
        return get_sql_chain()
    else:
        print("Routing to general_chain")
        return create_general_chain()


router_chain = (
    {
        "topic": chain,  # chain 실행 결과를 적절히 변환
        "question": itemgetter("question"),
        # "question": RunnablePassthrough(),
        "chat_history": itemgetter("chat_history"),
    }
    | RunnableLambda(route)
    | StrOutputParser()  # 필요 없을듯 여기서 공통 LLM을 여기서 넣어야할듯
)

route_with_history = RunnableWithMessageHistory(
    router_chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)


def get_router_chain():
    st.session_state["router_chain"] = route_with_history
    return router_chain
