# Imports
import streamlit as st
import pandas as pd
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

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
        OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
        MODEL = st.sidebar.selectbox("Select Model", options=["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"])

    # Upload File
    file = st.file_uploader("Upload CSV file",type=["csv"])
    if not file or not OPENAI_API_KEY: st.stop()

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
            llm = ChatOpenAI(temperature=TEMP, callbacks=[StreamHandler(st.empty())], streaming=True, openai_api_key=OPENAI_API_KEY, model=MODEL)

            # Define pandas df agent
            agent = create_pandas_dataframe_agent(llm, data, verbose=True, agent_type='openai-tools', allow_dangerous_code=True) 
            response = agent.invoke(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response['output']))

  
if __name__ == "__main__":
    main()   