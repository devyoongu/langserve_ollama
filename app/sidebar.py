import streamlit as st
import requests


def render_sidebar():
    """사이드바 렌더링 함수"""
    with st.sidebar:
        # 초기화 버튼 생성
        clear_btn = st.button("대화 초기화")

        # 파일 업로드
        uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

        # # 모델 선택 메뉴
        # selected_model = st.selectbox("LLM 선택", ["ollama", "xionic"], index=0)

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

        # 초기화 버튼이 눌리면...
        if clear_btn:
            st.session_state["messages"] = []
