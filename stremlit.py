import streamlit as st
from rag import run_rag
from embedding import load_pdf
import io
# Initialize the LLM (example with OpenAI)

# Streamlit App Layout
# Streamlit App Layout
st.title("LLM Question Answering with Reference Document")

# Input Box for User Question
user_question = st.text_input("Enter your question:", value="", max_chars=500)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
if uploaded_file is not None:
    byte_data = uploaded_file.read()
    # Convert byte data to a file-like object using io.BytesIO
    pdf_file = io.BytesIO(byte_data)
    
    # Extract text from the uploaded PDF
    status: str =  load_pdf(pdf_file, uploaded_file.name)
    st.success(f"{status}") 

    # Display the extracted text

# Button to Trigger Prediction
if st.button("Get Answer"):
    if user_question.strip():
        try:
            # Use the LLM to predict an answer based on the user question and reference document
            response = run_rag(user_question, uploaded_file.name)
            
            # Display the result
            st.success("Answer:") 
            if "Page Content:" in response:
                st.write(response.split("Page Content:")[0].strip())
                st.text_area("Page Reference Document:", response.split("Page Content:")[0].strip(), height=200, max_chars=1000)
            elif "Document:" in response:
                st.write(response.split("Document:")[0].strip())
                st.text_area("document Reference Document:", response.split("Document:")[1].strip(), height=200, max_chars=1000)
            else:
                st.write(response.splitlines()[0])
                #st.write("No recognizable content to split.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question before clicking the button.")
