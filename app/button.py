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
        if st.button("어떤 탁구채를 사용해?"):
            selected_category = "어떤 탁구채를 사용해?"
    with col2:
        if st.button("we톡에 대해 알려줘"):
            selected_category = "we톡에 대해 알려줘"
    with col3:
        if st.button("대한민국의 수도는 어디야?"):
            selected_category = "대한민국의 수도는 어디야?"

    return selected_category
