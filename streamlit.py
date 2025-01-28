import streamlit as st
from rag import run_rag
from embedding import load_pdf
import io
import time

# Simulate typing effect by gradually displaying text
def typing_effect(text: str, delay: float = 0.05):
    """Simulate typing effect in Streamlit."""
    placeholder = st.empty()  # Create a placeholder in Streamlit
    display_text = ""
    for char in text:
        display_text += char  # Add one character at a time
        placeholder.text(display_text)  # Update the placeholder with the new text
        time.sleep(delay)  # Add delay between each character to simulate typing

# Streamlit App Layout
st.title("LLM Question Answering with Reference Document")


# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


if uploaded_file is not None:
    try:
        # Read and process the uploaded PDF
        byte_data = uploaded_file.read()
        pdf_file = io.BytesIO(byte_data)
        
        # Extract text from the uploaded PDF
        status: str = load_pdf(pdf_file, uploaded_file.name)
        st.success(f"{status}")
    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e}")
        

# Input Box for User Question
user_question = st.text_input("Enter your question:", value="", max_chars=500)

# Button to Trigger Prediction
if st.button("Get Answer"):
    if user_question.strip():
        if uploaded_file is None:
            st.warning("Please upload a PDF file before asking a question.")
        else:
            try:
                # Use the LLM to predict an answer based on the user question and reference document
                response = run_rag(user_question, uploaded_file.name)
                
                # Display the result
                st.success("Answer:")
                if "Page Content:" in response:
                    content, reference = response.split("Page Content:", 1)
                    typing_effect(content.strip())
                    st.text_area("Page Reference Document:", reference.strip(), height=200, max_chars=1000)
                elif "Document:" in response:
                    content, reference = response.split("Document:", 1)
                    typing_effect(content.strip())
                    st.text_area("Document Reference Document:", reference.strip(), height=200, max_chars=1000)
                else:
                    typing_effect(response.strip())
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question before clicking the button.")
