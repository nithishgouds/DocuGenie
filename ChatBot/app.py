import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["PYTORCH_NO_LINT_WARN"] = "1"

import sys
import types
sys.modules['torch.classes'] = types.ModuleType("torch.classes")

import streamlit as st
from dotenv import load_dotenv
import traceback
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from HTMLTemplates import css, user_template, bot_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline


def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error("Please upload and process your PDFs first!")
        return

    formatted_question = (
    f"Carefully read the provided document context and answer the following question in detail and in a clean, readable manner.\n\n"
    f"Question: {user_question}\n\n"
    f"Answer:"
)

    print("[DEBUG] User Question:", formatted_question)

    try:
        response = st.session_state.conversation.invoke(
            {'question': formatted_question},
            config={"configurable": {"session_id": "user-session"}}
        )
        print("[DEBUG] Model response:", response)

        st.session_state.chat_history.append(type('msg', (object,), {
            'type': 'human',
            'content': user_question
        })())

        st.session_state.chat_history.append(type('msg', (object,), {
            'type': 'ai',
            'content': response.get('answer', 'No response generated.')
        })())

    except Exception as e:
        st.error("Error occurred while generating a response.")
        st.code(traceback.format_exc())


def get_conversation_chain(vectorstore):
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        truncation=True
    )

    llm = HuggingFacePipeline(pipeline=gen_pipeline)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    runnable = RunnableWithMessageHistory(
        chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return runnable


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[DEBUG] Number of text chunks for vectorstore:", len(text_chunks))
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print("[DEBUG] Sample chunk preview:", chunks[:2])
    return chunks


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    print("[DEBUG] Extracted PDF text (first 500 chars):", text[:500])
    return text


def main():
    try:
        load_dotenv()
        st.set_page_config(page_title="DocuGenie: Ask Your PDFs Anything ✨")
        st.write(css, unsafe_allow_html=True)

        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.header("Chat with multiple PDF :books:")

        user_question = st.text_input("Ask a question about your documents")
        if user_question:
            print("[DEBUG] Text input received.")
            handle_userinput(user_question)

            

        if st.session_state.chat_history:
            for msg in st.session_state.chat_history:
                if msg.type == "human":
                    st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("""
    <h1 style='font-size:38px; font-weight: bold; color: white; display: flex; align-items: center; margin-bottom: 10px;'>
        🧞‍♂️&nbsp; DocuGenie
    </h1>
""", unsafe_allow_html=True)


            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Process'", accept_multiple_files=True)

            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.warning("No text could be extracted from the uploaded PDFs.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store)


    except Exception as e:
        st.error("An error occurred. See console for details.")
        traceback.print_exc()


if __name__ == '__main__':
    main()
