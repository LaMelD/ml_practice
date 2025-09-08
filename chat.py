import streamlit as st
import uuid
from dotenv import load_dotenv

from llm import get_ai_response

st.set_page_config(page_title="챗봇 - TEST", page_icon="🤖")

st.title("🤖 챗봇 - TEST")

st.caption("물어보신 내용을 AI가 답변해드립니다.")

load_dotenv()

# 세션마다 고유 ID 부여
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

#print("현재 세션 ID:", st.session_state["session_id"])

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="여기에 질문을 입력하세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question, st.session_state["session_id"])
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response) 
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
