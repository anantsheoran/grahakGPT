# grahakGPT: AI-Powered Customer Support Chatbot

"Grahak" (ग्राहक) is the Hindi word for **Customer**.

**grahakGPT** is a Retrieval-Augmented Generation (RAG) system designed to provide accurate, instant, and context-aware customer support. By using a pre-defined knowledge base, the application ensures that the AI's responses are grounded in your specific company or product information, reducing hallucinations and improving response quality.

[![Deployed on Vercel](https://vercel.com/button)](https://grahak-gpt.vercel.app)

## ✨ Features

* **Context-Aware Responses:** Utilizes a custom `knowledge_base.txt` file to ground LLM responses, ensuring answers are specific and relevant.
* **Retrieval-Augmented Generation (RAG):** Implements a RAG pipeline to retrieve relevant information before generating a final answer.
* **Simple Knowledge Update:** Easily update the bot's knowledge by modifying the plain text file.
* **Modular Architecture:** Clear separation of concerns with dedicated files for the frontend, backend logic, and knowledge base processing.
* **Easy Deployment:** Designed for seamless deployment on platforms like Vercel.

## 🚀 Live Demo

You can interact with a live deployment of the project here:

[**grahak-gpt.vercel.app**](https://grahak-gpt.vercel.app)

## 🛠️ Installation and Setup

### Prerequisites

1.  **Python 3.x**
2.  An **OpenAI API Key** (or API Key for the LLM provider you choose to integrate).

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/anantsheoran/grahakGPT.git](https://github.com/anantsheoran/grahakGPT.git)
    cd grahakGPT
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```

3.  **Install Dependencies:**

    The necessary libraries are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Environment Variables:**

    Create a file named `.env` in the root directory and add your API key.

    ```
    # .env
    OPENAI_API_KEY="YOUR_API_KEY_HERE"
    ```

5.  **Configure Knowledge Base:**

    Edit the `knowledge_base.txt` file to include all the customer support documentation, FAQs, or product information you want the bot to use. Ensure the content is structured logically for better retrieval.

## 💻 Usage

### Running Locally

To run the application, you will typically execute the frontend file (or a main application file that initializes the server).

```bash
# Depending on your setup (e.g., if using Streamlit or a specific web framework):
python frontend.py
