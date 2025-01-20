import streamlit as st


def render_buttons():
    """
    버튼 생성 로직을 처리하는 함수.
    Returns:
        str: 선택된 카테고리 텍스트 (None if no button is clicked)
    """
    selected_category = None
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("배상책임 담당자 연락처를 알려줘"):
            selected_category = "배상책임 담당자 연락처를 알려줘"
    with col2:
        if st.button("민사소송 담당자 연락처 알려줘"):
            selected_category = "민사소송 담당자 연락처 알려줘"
    with col3:
        if st.button("아파트 현관 앞에서 넘어져서 팔과 대퇴골이 골절되었습니다."):
            selected_category = (
                "아파트 현관 앞에서 넘어져서 팔과 대퇴골이 골절되었습니다."
            )
    with col4:
        if st.button(
            "마트에서 장을 보다가 미끄러져 넘어지면서 팔이 골절된 경우 보상 여부는?"
        ):
            selected_category = (
                "마트에서 장을 보다가 미끄러져 넘어지면서 팔이 골절된 경우 보상 여부는?"
            )

    return selected_category
