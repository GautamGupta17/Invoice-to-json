import streamlit as st
import json
from io import BytesIO
from utils import process_pdf, process_image

# Streamlit UI
st.title("Swipe Intern Hiring - Invoice Parser for Multiple Filetypes", anchor="top")
st.write("Upload a PDF or PNG/JPG file to process:")

# Move file uploader to sidebar
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "png", "jpg"])

if uploaded_file is not None:
    if isinstance(uploaded_file, BytesIO):
        file_type = uploaded_file.type.split("/")[1]
    else:
        file_type = uploaded_file.type.split("/")[1]
        uploaded_file = BytesIO(uploaded_file.read())

    st.write(f"File type detected: {file_type}")
    
    try:
        if file_type == "pdf":
            result = process_pdf(uploaded_file)
        elif file_type in ["png", "jpg"]:
            result = process_image(uploaded_file)
        else:
            st.error("Unsupported file type!")
            result = None

        if result is not None:
            # Format the result to handle new lines
            formatted_result = result.replace('\n', '<br>')
            st.markdown(f"**Result:**<br>{formatted_result}", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
