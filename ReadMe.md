\# 🏡 Luxury Home Decor Assistant (RAG-based with DeBERTa + LLaMA-2)

This is a Retrieval-Augmented Generation (RAG) project that helps users
discover luxury home decor ideas based on real product descriptions. It
combines the power of \*\*DeBERTa\*\* for semantic embedding,
\*\*FAISS\*\* for fast similarity search, and \*\*LLaMA-2\*\* for
natural language generation.

\-\--

\## 💡 Features

\- 🔍 \*\*Semantic Search\*\* using FAISS over product descriptions  -
🤖 \*\*RAG-based Answering\*\* using LLaMA-2 LLM  - 💬 \*\*Gradio
Interface\*\* for real-time question-answering  - 📦 Packaged for easy
deployment on Hugging Face Spaces

\-\--

\## 🚀 How It Works

1\. \*\*User inputs a question\*\*, e.g., \_\"What are some luxury home
decor ideas?\"\_ 2. \*\*DeBERTa\*\* generates an embedding for the query
3. \*\*FAISS\*\* retrieves top-k relevant product descriptions from the
dataset 4. \*\*LLaMA-2\*\* generates a detailed, context-aware response
using those descriptions

\-\--

\## 📂 Project Structure

OnlineRetail/ ├── app.py \# Gradio app entry point ├── requirements.txt
\# Python dependencies ├── deberta_text_data.csv \# Cleaned product
descriptions ├── deberta_faiss.index \# FAISS similarity index

\- \*\*Embedding Model\*\*: \`microsoft/deberta-v3-base\`  - \*\*LLM
Generator\*\*: \`meta-llama/Llama-2-7b-chat-hf\`  - \*\*Vector
Index\*\*: FAISS (Cosine Similarity)

\-\--

\## 🛠️ Run Locally

\`\`\`bash pip install -r requirements.txt python app.py

Deploy on Hugging Face Spaces Zip the folder

Upload to your Hugging Face Space

Select Gradio as the SDK

Dataset Source The product data is based on the Online Retail Dataset,
preprocessed and cleaned for relevance and consistency.

Author Dinesh Kumar Data Scientist \| Technical Project Manager \| NLP
Enthusiast
