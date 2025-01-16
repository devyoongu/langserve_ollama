from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from ragChain import get_route_rag_chain
from memoryChain import create_memory_chain
from operator import itemgetter

# Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", stream=True)
llm = ChatOpenAI(model="gpt-3.5-turbo", stream=True)

prompt = load_prompt("prompts/00_route.yaml", encoding="utf-8")


# 체인을 생성합니다.
chain = prompt | llm | StrOutputParser()  # 문자열 출력 파서를 사용합니다.

math_chain = (
    PromptTemplate.from_template(
        """You are an expert in math. \
Always answer questions starting with "깨봉선생님께서 말씀하시기를..". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    # OpenAI의 LLM을 사용합니다.
    | llm
)

science_chain = (
    PromptTemplate.from_template(
        """You are an expert in science. \
Always answer questions starting with "법무법인에서 말씀 드리겠습니다...". \
Respond to the following question:

Question: {question}
Answer:"""
    )
    # OpenAI의 LLM을 사용합니다.
    | llm
)

general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question concisely:

Question: {question}
Answer:"""
    )
    # OpenAI의 LLM을 사용합니다.
    | llm
)


def route(info):
    # 입력 데이터 형식 검증
    if not isinstance(info, dict):
        print(f"Invalid input for route: {info}")
        raise ValueError("Input to route function must be a dictionary.")

    if "탁구" in info["topic"].lower():
        print("Routing to rag_chain from session_state (탁구 관련)")
        # return get_route_rag_chain()
        return create_memory_chain()
    elif "법무" in info["topic"].lower():
        print("Routing to law science_chain")
        return science_chain
    else:
        print("Routing to general_chain")
        return general_chain


router_chain = (
    {
        "topic": chain,  # chain 실행 결과를 적절히 변환
        "question": itemgetter("question"),
        # "question": RunnablePassthrough(),
    }
    | RunnableLambda(route)
    | StrOutputParser()
)


def get_router_chain():
    st.session_state["router_chain"] = router_chain
    return router_chain
