import streamlit as st


def render_buttons():
    """
    버튼 생성 로직을 처리하는 함수.
    Returns:
        str: 선택된 카테고리 텍스트 (None if no button is clicked)
    """
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

    return selected_category
