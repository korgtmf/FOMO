import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model_id = "korgtmf/FOMO"  # your fine-tuned model on HF
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

st.title("FOMO – Financial News Topic/Sentiment Demo")

text = st.text_area("Enter financial news / tweet:")

if st.button("Predict") and text.strip():
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))

    st.write(f"Predicted class id: {pred_id}")
    st.bar_chart(probs.numpy())
