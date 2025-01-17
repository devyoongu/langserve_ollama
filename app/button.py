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
        if st.button("어떤 탁구채를 사용해?"):
            selected_category = "어떤 탁구채를 사용해?"
    with col2:
        if st.button("직원 리스트를 알려줘"):
            selected_category = "직원 리스트를 알려줘"
    with col3:
        if st.button("안녕 내 이름은 테디야"):
            selected_category = "안녕 내 이름은 테디야"
    with col4:
        if st.button("내 이름이 뭐라고?"):
            selected_category = "내 이름이 뭐라고?"

    return selected_category
