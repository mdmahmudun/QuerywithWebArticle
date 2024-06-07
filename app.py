import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def get_text_from_urls(urls):
    loader = UnstructuredURLLoader(urls = urls)
    data = loader.load()
    return data
def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_documents(data)
    return chunks
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = st.session_state.vector_store.as_retriever())
    return chain

def user_input(user_question, chat_history):
    chain = get_conversational_chain()
    results = chain({"question": user_question}, return_only_outputs = True)
    bot_text = results['answer'] + '\n\n' + "Source: " + results.get("sources","")
    chat_history.append({"user": user_question, "bot": bot_text})
    return chat_history

def display_chat(chat_history):
    for chat in chat_history:
        st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='background-color: #ffcccb; border-radius: 50%; padding: 10px; margin-right: 10px;'>User</div>
                <div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; flex: 1;'>{chat['user']}</div>
            </div>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='background-color: #add8e6; border-radius: 50%; padding: 10px; margin-right: 10px;'>Bot</div>
                <div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; flex: 1;'>{chat['bot']}</div>
            </div>
        """, unsafe_allow_html=True)





def main():
    st.set_page_config('Chat with Web Articles')
    st.header("Chat with Web Articles")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = []
    

    user_question = st.chat_input("Ask a Question From the Given Links")
    
    if user_question:
        st.session_state.chat_history = user_input(user_question=user_question, chat_history=st.session_state.chat_history)

    display_chat(st.session_state.chat_history)
    with st.sidebar:
        st.title('Menu')
        num_urls = st.number_input("Enter Number of Links:")
        num_urls = int(num_urls)
        urls = []
        for i in range(num_urls):
            url = st.sidebar.text_input(f"Enter URL {i+1}")
            urls.append(url)
        if st.button('Submit & Process', key = 'process'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Getting Texts form URLs...")
            raw_text = get_text_from_urls(urls)
            progress_bar.progress(33)
            status_text.text("Making Chunks from Texts...")
            text_chunks = get_text_chunks(raw_text)
            progress_bar.progress(66)
            status_text.text("Creating Vector Store...")
            st.session_state.vector_store = get_vector_store(text_chunks)
            progress_bar.progress(100)
            status_text.text("Done!")
            st.success('Done')


if __name__ == "__main__":
    main()

