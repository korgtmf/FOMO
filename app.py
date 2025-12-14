import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Topic id -> label mapping (20 financial topics)
TOPIC_ID2LABEL = {
    0: "Analyst Update",
    1: "Fed | Central Banks",
    2: "Company | Product News",
    3: "Treasuries | Corporate Debt",
    4: "Dividend",
    5: "Earnings",
    6: "Energy | Oil",
    7: "Financials",
    8: "Currencies",
    9: "Politics",
    10: "M&A | Investments",
    11: "Markets",
    12: "Macro",
    13: "Tech",
    14: "Commodities",
    15: "Fixed Income",
    16: "Economy",
    17: "Real Estate",
    18: "Metals",
    19: "Legal | Regulation",
}

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

    # Get the predicted label based on the predicted ID
    pred_label = TOPIC_ID2LABEL.get(pred_id, f"Unknown ({pred_id})")

    # Display only the predicted topic label
    st.write(f"Predicted topic: {pred_label}")

    # Show the probabilities in a bar chart
    st.bar_chart(probs.numpy())
