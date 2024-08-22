import streamlit as st
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import os

def main():
    load_dotenv()
     # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV ")
    st.header("Ask your CSV")
    
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        user_question = st.text_input("Ask a question about your CSV: ")
        llm=OpenAI(temperature=0)

        agent = create_csv_agent(
            llm, csv_file, verbose=True,allow_dangerous_code=True)

        #user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            response=agent.run(user_question)
            #with st.spinner(text="In progress..."):
                #st.write(agent.run(user_question))
            st.write(response)
if __name__=="__main__":
    main()