# Imports
import streamlit as st
import pandas as pd
from langchain_core.messages import ChatMessage
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.agent_toolkits import create_sql_agent

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def main():
    st.set_page_config(page_title="Chat with CSV", page_icon="📊")
    st.subheader("Chat with CSV")
    st.write("Upload a CSV file and query answers from your data.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

    # Define settings
    with st.sidebar:
        TEMP = st.slider(label="LLM Temperature", min_value=0.0, max_value=1.0, value=0.5)
        st.button("Clear Messages", on_click=lambda: st.session_state.update(messages=[ChatMessage(role="assistant", content="How can I help you?")]))

    # Upload File
    file = st.file_uploader("Upload CSV file",type=["csv"])

    instructions = st.text_area("Instructions", placeholder="Enter instructions for the chatbot")
    
    if not file: st.stop()
    # Read Data as Pandas
    data = pd.read_csv(file)

    # Display Data Head
    st.write("Data Preview:")
    st.dataframe(data) 

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # Define large language model (LLM)
            llm = OllamaFunctions(model="gemma2:27b", temperature=TEMP, callbacks=[StreamHandler(st.empty())], streaming=True)
            
            # Define SQL Database
            engine = create_engine('sqlite:///:memory:')
            data.to_sql('data', engine, index=False)
            db = SQLDatabase(engine=engine)

            # Define pandas df agent
            agent = create_sql_agent(llm, db=db, verbose=True, agent_type='openai-tools')
            if instructions:
                messages = [ChatMessage(role="user", content=instructions)] + st.session_state.messages
                print(messages)
                response = agent.invoke(messages)
            else:
                response = agent.invoke(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response['output']))

  
if __name__ == "__main__":
    main()