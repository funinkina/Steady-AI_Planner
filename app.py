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
import time

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

EMBEDDINGS_FILE = "embeddings.csv"  # Define your embeddings file path
CHROMA_DB_DIR = "chroma_db"  # Directory to store ChromaDB files
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


def process_pdf(in_memory_file, embedding_model):
    # First it will open the PDF file and then process it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        in_memory_file.seek(0)
        temp_pdf.write(in_memory_file.read())
        temp_pdf_path = temp_pdf.name

    try:
        # Load the PDF file and split it into pages
        pdf_loader = PyPDFLoader(temp_pdf_path)
        text_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=250,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        document_pages = pdf_loader.load_and_split(text_splitter)

        # if os.path.exists(EMBEDDINGS_FILE):
        #     # Load embeddings from file
        #     with open(EMBEDDINGS_FILE, "r") as csv_file:
        #         csv_reader = csv.reader(csv_file)
        #         loaded_embeddings = [list(map(float, row)) for row in csv_reader]
        #     vector_db = Chroma.from_documents(document_pages, embedding_model, embeddings=loaded_embeddings)
        # else:
        # Compute embeddings and save to file
        vector_db = Chroma.from_documents(document_pages, embedding_model)
        computed_embeddings = vector_db.embeddings
        with open(EMBEDDINGS_FILE, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(computed_embeddings)

        retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    finally:
        os.remove(temp_pdf_path)

    return retriever


def display_timer():
    if st.session_state.timer_running:
        # Calculate minutes and seconds
        minutes, seconds = divmod(st.session_state.time_left, 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        # Display the timer with the current phase
        st.write(f"### {st.session_state.current_phase.capitalize()} Time: {time_str}")

        # Countdown
        st.session_state.time_left -= 1

        if st.session_state.time_left < 0:
            if st.session_state.current_phase == 'productivity':
                # Switch to break time
                st.session_state.current_phase = 'break'
                st.session_state.time_left = 10 * 60  # 10 minutes in seconds for break phase
            else:
                # Switch to productivity time
                st.session_state.current_phase = 'productivity'
                st.session_state.time_left = 20 * 60  # 20 minutes in seconds for productivity phase

        # Check if total duration is over
        if st.session_state.total_duration <= 0:
            st.session_state.timer_running = False
            st.write("## Time's up! Great job!")
        else:
            st.session_state.total_duration -= 1

        # Refresh the page every second to update the countdown
        time.sleep(1)
        st.experimental_rerun()


def timer_window(duration):
    duration = int(duration.split()[0])
    if duration == 1:
        study_session = 20 * 60 * 60
        break_session = 10 * 60 * 60
        for i in range(2):
            with st.empty():
                while study_session:
                    minutes, secs = divmod(study_session, 60)
                    time_now = '{:02d}:{:02d}'.format(minutes, secs)
                    st.header(f"{time_now}")
                    time.sleep(1)
                    study_session -= 1
            st.header("Good Study session, break starts now")
            with st.empty():
                while break_session:
                    minutes, secs = divmod(study_session, 60)
                    time_now = '{:02d}:{:02d}'.format(minutes, secs)
                    st.header(f"{time_now}")
                    time.sleep(1)
                    study_session -= 1
            st.header("Enough Break, time to study now!!")


def answer_ui(retriever, type_adhd, task, duration):

    user_input = f"""
    i have {type_adhd} adhd. give me a study plan to finish {task} in {duration}.
    suggest me tips on how i should study, how long and frequent should my break be. 
    give me a table with the time schedule. time durations should be in the first row
    and task should be on the second in markdown format.
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
                if st.button("Ready to begin?"):
                    timer_window(duration)
            except Exception as e:
                st.error(f"Error getting response: {e}")


def main():
    # load_custom_css()
    st.title("Steady AI Planner")

    # Open the PDF file
    with open("document.pdf", "rb") as f:
        uploaded_file = f.read()

    with st.spinner("Processing request"):
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
