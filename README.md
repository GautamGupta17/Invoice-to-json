# STREAMLIT APP: RAG Based Invoice-to-json
Streamlit App: RAG-Based PDF and Image Invoices to JSON within seconds with extreme accuracy. 

## Overview

This Streamlit app demonstrates a Retrieval-Augmented Generation (RAG) system for processing and analyzing PDF and image files. The app uses the Mistral-7B-v0.1 model and also gives support for with the OpenAI API to provide a chat-based interface for querying information extracted from these files.

## Problem Statement

With the increasing volume of digital documents, it becomes challenging to efficiently extract and analyze text from various file formats such as PDFs and images. This app addresses the need for a user-friendly interface to upload, process, and query information from these files, leveraging advanced language models to enhance text retrieval and analysis.

## Solution

This app provides a solution by:

1. Allowing users to upload PDF and image files.
2. Processing the files to extract text using LangChain and PyPDFLoader for PDFs, and image processing functions for images.
3. Storing the extracted information in a structured format using ChromaDB.
4. Giving the output as a json string

## Demo

<table>
  <tr>
    <td><img src="https://github.com/GautamGupta17/Invoice-to-json/blob/main/demo/1_image.png" alt="Demo Image 1" width="300"/></td>
    <td><img src="https://github.com/GautamGupta17/Invoice-to-json/blob/main/demo/1_pdf.png" alt="Demo Image 2" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/GautamGupta17/Invoice-to-json/blob/main/demo/2_image.png" alt="Demo Image 3" width="300"/></td>
    <td><img src="https://github.com/GautamGupta17/Invoice-to-json/blob/main/demo/2_pdf.png" alt="Demo Image 4" width="300"/></td>
  </tr>
</table>


You can see the app in action with the demo images above. The app processes the uploaded files and displays the extracted information with extreme precision, using only open source solutions.

## Steps to Run the App


### 1. Clone the Repository

To start, clone the repository to your local machine. Open your terminal and run:

    ```bash
    git https://github.com/GautamGupta17/Invoice-to-json.git
    ```

Then, navigate into the cloned directory:

    ```bash
    cd Invoice-to-json
    ```

### 2. Install Dependencies

Ensure that you have Python installed. Install the required Python packages listed in the `requirements.txt` file by running:

    ```bash
    pip install -r requirements.txt
    ```

This will set up all the necessary libraries and dependencies for the app.

### 3. Run the Streamlit App

With the dependencies installed, you can now start the Streamlit app. Execute the following command:

    ```bash
    streamlit run app.py --server.enableXsrfProtection false
    ```

This command will launch the Streamlit app in your default web browser and allow you to upload files

### 4. Upload Files and Interact

Once the app is running, it will be accessible in your browser at `http://localhost:8501` (or another port if specified).

- Use the sidebar to upload a PDF or image file.
- Get the json formatted query from the invoice
- For retrieving extra information, update the prompt in app.py

Enjoy using the app!
