import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory


def get_session_history(session_id):
    print(f"Received session_id: {session_id}")

    if "store" not in st.session_state:
        print("Initializing session_state store")
        st.session_state["store"] = {}

    if session_id not in st.session_state["store"]:
        print(
            f"Session ID '{session_id}' not found in store, initializing new ChatMessageHistory."
        )
        st.session_state["store"][session_id] = ChatMessageHistory()

        return st.session_state["store"][session_id]
    else:
        storeValue = st.session_state["store"][session_id]
        print(f"Session ID '{session_id}' found in store.")
        print(f"Stored sessions: {storeValue}")
        return storeValue
