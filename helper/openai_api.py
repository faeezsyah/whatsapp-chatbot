import os


import openai
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory


embeddings = OpenAIEmbeddings()
loader = PyPDFLoader('AAAA.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings)
memory = ConversationSummaryBufferMemory(memory_key="chat_history", return_messages=True, max_token_limit= 300)

def text_complition(prompt: str) -> dict:
    '''
    Call Openai API for text completion

    Parameters:
        - prompt: user query (str)

    Returns:
        - dict
    '''
    try:
        qa = ConversationalRetrievalChain.from_llm(
            OpenAI(model_name ='text-davinci-003' ,temperature=0.1,), 
            store.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        result1 = qa({"question": prompt})
        return {
            'status': 1,
            'response': result1['answer'][0]['text']
        }
    except:
        return {
            'status': 0,
            'response': ''
        }
