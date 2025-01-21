import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os, base64

st.title("Fashion Recommendation Bot")

openai_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o", api_key=openai_key)

if image := st.file_uploader("본인의 전신이 보이는 사진을 올려주세요!", type=['png', 'jpg', 'jpeg']):
  st.image(image)
  image = base64.b64encode(image.read()).decode("utf-8")
  with st.chat_message("assistant"):
    message = HumanMessage(
      content=[
        {"type": "text", "text": "사람의 전신이 찍혀있는 사진이 한 장 주어집니다. 이 때, 사진 속 사람과 어울리는 옷 및 패션 스타일을 추천해주세요."},
        {
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        },
      ],
    )
    result = model.invoke([message])
    response = result.content
    st.markdown(response)

# if "messages" not in st.session_state:
#   st.session_state.messages = []
    
# for message in st.session_state.messages:
#   with st.chat_message(message["role"]):
#     st.markdown(message["content"])
    
# if prompt := st.chat_input("What is up?"):
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("assistant"):
#         messages = []
#         for m in st.session_state.messages:
#             if m["role"] == "user":
#                 messages.append(HumanMessage(content=m["content"]))
#             else:
#                 messages.append(AIMessage(content=m["content"]))

#         result = model.invoke(messages)
#         response = result.content
        
#         st.markdown(response)

#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": response
#         })