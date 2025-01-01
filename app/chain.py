from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote.prompts import load_prompt


# retriever 를 통해 검색된 document_list는 메타정보까지 모두 포함되어 있기 때문에 format_doc을 체이닝으로 추가하여 content 내용만 연결된 String 값으로 리턴
# 불필요한 메터정보를 제외한 content 내용만 붙여서 yaml 템플릿의 context에 넣기 좋기 변환해주는 역할
def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성
def create_first_chain(retriever):
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
