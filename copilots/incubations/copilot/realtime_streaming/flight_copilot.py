import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from flight_copilot_utils import PERSONA, AVAILABLE_FUNCTIONS, FUNCTIONS_SPEC
import sys
sys.path.append("..")
from utils import Smart_Agent,add_to_cache
import time
import random
import os
from pathlib import Path  
import json
with open('./data/user_profile.json') as f:
    user_profile = json.load(f)
print(user_profile)
agent = Smart_Agent(persona=PERSONA.format(customer_name =user_profile['name'], customer_id=user_profile['customer_id']),functions_list=AVAILABLE_FUNCTIONS, functions_spec=FUNCTIONS_SPEC, init_message=f"Hi {user_profile['name']}, this is Maya, your airline customer specialist, what can I do for you?")

st.set_page_config(layout="wide",page_title="Real timeCopilot- A demo of Copilot application using GPT on top of real time data")
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


MAX_HIST= 5
# Sidebar contents
with st.sidebar:
    st.title('Flight Copilot')
    st.markdown('''
    Real timeCopilot- A demo of Copilot application using GPT on top of real time data.

    ''')
    add_vertical_space(5)
    st.write('Created by James N')
    if st.button('Clear Chat'):

        if 'history' in st.session_state:
            st.session_state['history'] = []

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'input' not in st.session_state:
        st.session_state['input'] = ""


user_input= st.chat_input("You:")

## Conditional display of AI generated responses as a function of user provided prompts
history = st.session_state['history']
      
if len(history) > 0:
    for message in history:
        if message.get("role") != "system" and message.get("name") is  None:
            with st.chat_message(message["role"]):
                    st.markdown(message["content"])
else:
    history, agent_response = agent.run(user_input=None)
    with st.chat_message("assistant"):
        st.markdown(agent_response)
    user_history=[]
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    stream_out, query_used, history, agent_response = agent.run(user_input=user_input, conversation=history, stream=True)
    with st.chat_message("assistant"):
        if stream_out:
            message_placeholder = st.empty()
            full_response = ""
            for response in agent_response:
                if len(response.choices)>0:
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            if query_used: #add to cache
                add_to_cache(query_used, full_response)
                print(f"query {query_used} added to cache")
            history.append({"role": "assistant", "content": full_response})
        else:
            st.markdown(agent_response)

st.session_state['history'] = history