import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os, base64

st.title("Fashion Recommendation Bot")

openai_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o", api_key=openai_key)

if 'image_urls' not in st.session_state:
  st.session_state.image_urls = []
  
if 'messages' not in st.session_state:
  st.session_state.messages = []

uploaded_files = st.file_uploader("본인의 전신이 보이는 사진을 올려주세요! (여러 장 업로드 가능)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

def upload_image(files) -> str:
  st.image([file for file in uploaded_files], caption=[f"업로드된 이미지 {i+1}" for i, file in enumerate(uploaded_files)], use_column_width=True)

  for image in files:
    image = base64.b64encode(image.read()).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{image}"
    
    if image_url not in st.session_state.image_urls:
      st.session_state.image_urls.append(image_url)

if uploaded_files:
  upload_image(uploaded_files)
  
for m in st.session_state.messages:
  with st.chat_message(m['role']):
    st.markdown(m['content'])
  
if prompot := st.chat_input("질문을 입력하세요", key="user_input"):
  with st.chat_message('user'):
    st.markdown(prompot)
  
  with st.chat_message('assistant'):
    messages = []
    for m in st.session_state.messages:
      role = m.get('role')
      if role == 'user':
        messages.append(HumanMessage(content=m['content']))
      elif role == 'assistant':
        messages.append(AIMessage(content=m['content']))
    
    cur_message = HumanMessage(
      content=[
        {"type": "text", "text": prompot}
      ]
    )
    
    for url in st.session_state.image_urls:
      cur_message.content.append({"type": "image_url", "image_url": { "url" : url}})
    
    st.session_state.messages.append({"role": "user", "content": prompot})
    
    messages.append(cur_message)
    result = model.invoke(messages)
    response = result.content
    
    st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})