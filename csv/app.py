import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
from utils import get_model_response


#Main app
def main():
    st.title("Chat with CSV using Gemini pro")

    #File Uploader
    uploaded_file=st.sidebar.file_uploader("Choose your CSV file",type="CSV")

    #Fetching path of the uploaded file
    if uploaded_file is not None:
        #use tempfile because CSVloader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False)as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path=tmp_file.name

            #Initializing CSV_Loader
            csv_loader=CSVLoader(file_path=tmp_file_path,encoding="utf-8",csv_args={
                'delimiter':','
            })

            #Load data into csv_loader
            data=csv_loader.load()

            #Initializing chat Interface
            user_input=st.text_input("Your Message:")
            print(user_input)

            if user_input:
                response=get_model_response(data,user_input)
                st.write(response)





if __name__=="__main__":
    main()
