import os
import gradio as gr
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Load environment secrets securely
login(token=os.getenv("HF_TOKEN"))  # Hugging Face login
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # OpenAI GPT-3.5 auth

# ✅ Load FAISS index and product descriptions
index = faiss.read_index("deberta_faiss.index")
text_data = pd.read_csv("deberta_text_data.csv")["Retrieved Text"].tolist()

# ✅ Load DeBERTa for embedding
deberta_model_name = "microsoft/deberta-v3-base"
deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
deberta_model = AutoModel.from_pretrained(deberta_model_name).to("cpu")

# ✅ Embedding function
def generate_embeddings(texts):
    tokens = deberta_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cpu")
    with torch.no_grad():
        embeddings = deberta_model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
    return embeddings

# ✅ RAG-based response using OpenAI
def generate_response(message, history):
    # Embed and retrieve context
    query_embedding = generate_embeddings([message])
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, k=5)
    retrieved_docs = [text_data[i] for i in indices[0]]
    context = "\n- " + "\n- ".join(set(retrieved_docs))

    # Format history for OpenAI
    chat_history = [{"role": "system", "content": "You're a helpful luxury interior design assistant."}]
    for user, bot in history:
        chat_history.append({"role": "user", "content": user})
        chat_history.append({"role": "assistant", "content": bot})

    # Append latest query
    chat_history.append({
        "role": "user",
        "content": f"Here are related product descriptions:{context}\n\nUser Question: {message}\n\nAnswer:"
    })

    # Call OpenAI GPT-3.5
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0.7,
        max_tokens=500
    )
    response = completion.choices[0].message.content.strip()

    # Generate a follow-up question
    followup_prompt = f"Based on this interior decor answer: '{response}', suggest a helpful follow-up question the user might ask next."
    followup_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": followup_prompt}],
        temperature=0.7,
        max_tokens=60
    )
    followup = followup_completion.choices[0].message.content.strip()

    final_output = f"🪑 **Decor Response:** {response}\n\n🔄 **Suggested Follow-Up:** {followup}"
    return final_output

# ✅ Gradio Chat UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🪑 Luxury Decor Assistant (RAG)")
    gr.Markdown("💬 Ask your interior design questions using real product descriptions. Powered by **DeBERTa + FAISS + OpenAI GPT-3.5**.")

    chatbot = gr.Chatbot(label="Chatbot")
    msg = gr.Textbox(label="Textbox", placeholder="e.g. Suggest cozy decor for a small bedroom", scale=8)
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        response = generate_response(message, chat_history)
        chat_history.append((message, response))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()