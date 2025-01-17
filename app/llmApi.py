import requests
import streamlit as st


# API 요청 함수
def send_chat_log_to_api(chat_logs):
    url = "http://localhost:8080/api/chat-log"
    payload = {
        "chatThreadId": st.session_state.get("chat_thread_id"),
        "chatLogs": chat_logs,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    if "data" in response_data and "id" in response_data["data"]:
        chat_thread_id = response_data["data"]["id"]
        st.session_state["chat_thread_id"] = response_data["data"]["id"]


def call_external_api(sqlQuery):
    """외부 API 호출."""
    api_url = "http://localhost:8080/api/sqldeck/execute"
    api_headers = {"Content-Type": "application/json"}
    api_body = {"sqlQuery": sqlQuery}
    try:
        api_response = requests.post(api_url, headers=api_headers, json=api_body)
        api_response.raise_for_status()
        return api_response.json()
    except requests.RequestException as e:
        st.chat_message("assistant").write(f"External API call failed: {str(e)}")
        return {"error": "API call failed"}
