import streamlit as st

st.title("ChatBot")

if "messages" not in st.session_state:
  st.session_state.messages = []
    
    
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    
if prompt := st.chat_input("What is up?"):  # User의 input을 기다립니다. Placeholder로 "What is up?"라는 문구를 사용합니다.
  with st.chat_message("user"):
    st.markdown(prompt)  # User의 메시지를 기록합니다.
  st.session_state.messages.append({"role": "user", "content": prompt})  # User의 메시지를 session_state에 저장합니다.

  response = f"Echo: {prompt}"  # 우리가 내놓을 답변으로 user가 보낸 메시지를 사용합니다.

  # User의 메시지를 처리하는 방식을 똑같이 사용합니다(UI에 기록 및 state에 저장)
  with st.chat_message("assistant"):
    st.markdown(response)
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response
    })