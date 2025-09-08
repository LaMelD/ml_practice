from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

store = {}

# 세션별 history 저장
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm(model="gemma3:4b"):
    return ChatOllama(model=model)

def get_rag_chain():
    llm = get_llm()
    # 단순 예시 prompt
    system_prompt = (
        "당신은 전문가 어시스턴트입니다. 대화 맥락을 고려해 답변하세요."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),   # history 반영
            ("human", "{input}"),
        ]
    )

    # documents chain 대신 간단히 llm만 연결
    qa_chain = qa_prompt | llm | StrOutputParser()

    # history-aware runnable로 감쌈
    conversational_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",   # 여기서는 'output'
    )

    return conversational_chain

def get_ai_response(user_message: str, session_id: str="abc123"):
    rag_chain = get_rag_chain()

    # 실행 (세션 아이디는 고정 예시)
    response = rag_chain.stream(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    return response