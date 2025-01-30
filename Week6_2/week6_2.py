import streamlit as st
from langchain_openai import ChatOpenAI
import core.utility
import os

st.title('GitHub Code Review')

openai_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o", api_key=openai_key)
  
if 'messages' not in st.session_state:
  st.session_state.messages = []
  
for m in st.session_state.messages:
  with st.chat_message(m['role']):
    st.markdown(m['content'])
  
if prompt := st.chat_input("질문을 입력하세요", key="user_input"):
  messages = []
  for m in st.session_state.messages:
    role = m.get('role')
    if role == 'user':
      messages.append(f"사용자: {m['content']}")
    elif role == 'assistant':
      messages.append(f"어시스턴트: {m['content']}")
  
  review_prompt = ''
  codes = core.utility.parse_codes(prompt)
  if codes:
    history = '\n'.join(messages)
    prev_codes = core.utility.search_prev_codes_from_vectordb(codes)
    review_prompt = core.utility.get_review_prompt(history=history, prev_codes=prev_codes, codes=codes)
    # code가 있다면 코드 리뷰로 판단하여 review prompt로 질의하도록 한다.

  with st.chat_message('user'):
    st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.append(f"사용자: {prompt}")

    
  with st.chat_message('assistant'):
    response = ''
    if review_prompt:
      result = model.invoke(review_prompt)
      response = result.content
      
      core.utility.save_to_vectordb(codes=codes)
    else:
      result = model.invoke(messages)
      response = result.content
    
    st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})