import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import csv
from io import BytesIO
import tempfile
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

EMBEDDINGS_FILE = "embeddings.csv"

# Load the models
llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Retrieval chain template
template = """
You are a helpful AI assistant.
You help to plan schedule with kids having ADHD and help then with useful tips.
give a long thorough answer and use markdown 
context: {context}
input: {input}
answer:
"""


def process_pdf(in_memory_file, embedding):
    # first it will open the pdf file and then it will process the pdf file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as dumpfile:
        in_memory_file.seek(0)
        dumpfile.write(in_memory_file.read())
        tempfilepath = dumpfile.name
    # Load the PDF file and split it into pages
    try:
        loader = PyPDFLoader(tempfilepath)
        text_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=250,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        pages = loader.load_and_split(text_splitter)

        # Check if embeddings file exists
        # if os.path.exists(EMBEDDINGS_FILE):
        #     # Load embeddings from file
        #     with open(EMBEDDINGS_FILE, "r") as f:
        #         reader = csv.reader(f)
        #         embeddings = [list(map(float, row)) for row in reader]
        # else:
        # Compute embeddings and save to file
        vectordb = Chroma.from_documents(pages, embedding)
        embedding = vectordb.embeddings
        # this thing is supposed to save the embeddings locally but its not working
        with open(EMBEDDINGS_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(embedding)

        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    finally:
        os.remove(tempfilepath)
    return retriever


def answer_ui(retriever, type_adhd, task, duration):

    user_input = f"""
    i have {type_adhd} adhd. give me a study plan to finish {task} in {duration}.
    suggest me tips on how i should study, how long and frequent should my break be. 
    """

    if user_input:
        with st.spinner("Finding solutions..."):
            try:
                prompt = PromptTemplate.from_template(template)
                # No change to your document chain creation as per your instruction
                combine_docs_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
                response = retrieval_chain.invoke({"input": user_input})
                st.markdown(
                    f"**Here are some tips to help you accomplish your goal:**\n\n{response['answer']}</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error getting response: {e}")


def main():
    # load_custom_css()
    st.title("Steady AI Planner")

    # Open the PDF file
    with open("document.pdf", "rb") as f:
        uploaded_file = f.read()

    with st.spinner("Processing upload..."):
        in_memory_file = BytesIO(uploaded_file)
        st.success("Data Loaded")

        try:
            retriever = process_pdf(in_memory_file, embeddings)
            st.success("Model Trained")
            st.subheader("Select Parameters:")
            col1, col2, col3 = st.columns(3)
            with col1:
                type_adhd = st.selectbox(
                    "Type of ADHD:",
                    ('Inattentive', 'Hyperactive Impulsive ', 'Combination type'),
                )
            with col2:
                task = st.selectbox(
                    "Task you want to finish:",
                    ('Study', 'Preparation/revision', 'Assignment', 'Project'),
                )
            with col3:
                duration = st.selectbox(
                    "The amount of duration:", ('1 Hour', '2 Hour', '3 Hour')
                )
            if st.button("Start Planning"):
                answer_ui(retriever, type_adhd, task, duration)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()
