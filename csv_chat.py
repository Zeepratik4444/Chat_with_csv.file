from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent,create_csv_agent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import pandas as pd
from langchain_openai import OpenAI
import os
import streamlit as st
import openai
from dotenv import load_dotenv
load_dotenv()


api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192")

st.set_page_config(page_title="Ask your CSV")
st.header("Ask your CSV 📈")

csv_file = st.file_uploader("Upload a CSV file", type="csv")
if csv_file is not None:
    df=pd.read_csv(csv_file)
    agent =create_pandas_dataframe_agent( llm, df, verbose=True, allow_dangerous_code=True)
    user_question = st.text_input("Ask a question about your CSV: ")
    if user_question is not None and user_question != "":
        with st.spinner(text="In progress..."):
            st.write(agent.run(user_question))