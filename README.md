# RAG System for Oracle CSS Internal Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based chatbot designed to assist the Oracle CSS team. It leverages both public and internal Oracle technology documentation, providing intelligent and context-aware responses to user queries.

## Architecture

**1. Data Preprocessing**  
- Implemented using OCI Data Science Notebooks.
- Authentication via Resource Principals for secure, keyless access.
- Utilizes the Oracle ADS SDK, as well as NLP libraries (such as `nltk`), for data cleaning and preparation.
- Preprocessed data is stored in OCI Block Storage.

**2. Local Chatbot Development**  
- The chatbot retrieves data from Block Storage using local configuration files (for development use).
- Embeddings and vector indices are generated locally.
- Embeddings are stored in Oracle Database 23ai for efficient retrieval.
- Responses are generated using a Large Language Model (LLM).
- The user interface is implemented with Streamlit for rapid prototyping and iteration.

## Getting Started

### Prerequisites

- Access to an Oracle Cloud Infrastructure (OCI) tenancy with necessary privileges (Data Science, Block Storage, Vault, DB 23ai).
- Local development environment with Python 3.8+.
- Required Python libraries:  
  ```
  oracle-ads
  oci
  oci[adk]
  nltk
  streamlit
  ```

### Installation

1. **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up OCI authentication:**
    - For OCI Data Science Notebooks, authentication is handled via Resource Principals (recommended).
    - For local development, create and securely store your `~/.oci/config` file. **Never commit credentials to version control.**

4. **Configure your environment:**
    - Update configuration files (e.g., database credentials, secret keys) as needed for your local or production environment.
    - For secret management, use OCI Vault whenever possible.

---

**Note:**  
Do not commit secrets or sensitive files to the repository. Always follow Oracle security, privacy, and compliance policies.

---