from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

genai.configure(api_key="AIzaSyDcs6FxHLMW4F2Iyd3icanLvY-MqxTW_Ok")
os.environ["GOOGLE_API_KEY"]="AIzaSyDcs6FxHLMW4F2Iyd3icanLvY-MqxTW_Ok"

#function to get response from GEMINI PRO
def get_model_response(file,query):

    #SPlt the context text into managable chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    context="\n\n".join(str(p.page_content) for p in file)

    data=text_splitter.split_text(context)

    embeddings=GoogleGenerativeAIEmbeddings(model='models/embeding.001')

    #Create embeddings for Data Chunks
    searcher=Chroma.from_texts(data,embeddings).as_retriever()

    #Fetch all vectors from the vector store ie searchers
    q="Which car brand name has maximum price?"
    records=searcher.get_relevant_documents(q)
    print(records)

    prompt_template="""
         You have to answer the questions from the provided context and make sure that you provide all the details\n
         Context:{context}\n
         Question:{question}\n
    
         Answer:
    
        """
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.9)

    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)

    response=chain(
        {
          "input_documents":records,
          "question":query
        }
        , return_only_output=True
    )

    return response['output_text']

