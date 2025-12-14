Library

Account

Upgrade
import os
from huggingface_hub import login
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate


# Login (better to set HF_TOKEN in environment)
os.environ["HF_TOKEN"] = "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" # remove in production
login(token=os.environ.get("HF_TOKEN"))


# Correct model for 20-class financial news topic classification
model_name = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Load dataset
dataset = load_dataset("zeroshot/twitter-financial-news-topic")


# Tokenize
def tokenize_function(examples):
return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Use small subsets for quick testing
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))


# Metrics
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
logits, labels = eval_pred
predictions = np.argmax(logits, axis=-1)
return accuracy_metric.compute(predictions=predictions, references=labels)


# Training arguments
training_args = TrainingArguments(
output_dir="test_trainer",
num_train_epochs=1,
eval_strategy="epoch",
per_device_train_batch_size=16,
per_device_eval_batch_size=16,
logging_dir='./logs',
)


trainer = Trainer(
model=model,
args=training_args,
train_dataset=small_train_dataset,
eval_dataset=small_train_dataset,
compute_metrics=compute_metrics,
)


# Evaluate (or train if you want)
print("Evaluating...")
trainer.evaluate()


# Optional: fine-tune for 1 epoch
# trainer.train()
This script is almost correct for quickly evaluating that model on the Twitter financial news topic dataset, but there are a few issues and risks to fix before running it in practice.​

Critical issues to fix
Hard‑coded HF token in code

Never embed an access token directly in source code, especially one that may be pushed to Git or shared. Use an environment variable or huggingface-cli login locally and remove the os.environ["HF_TOKEN"] = ... line entirely.​

After testing, go to your Hugging Face account and revoke this token, then create a new one with minimal required scopes.

Deprecated evaluation_strategy vs eval_strategy

In newer transformers versions, evaluation_strategy is being deprecated and replaced with eval_strategy.​

To avoid warnings or future errors, change the argument to:

python
training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=1,
    eval_strategy="epoch",  # instead of evaluation_strategy
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
)
Dataset and model compatibility
Model–dataset alignment

The model nickmuchi/finbert-tone-finetuned-finance-topic-classification is fine‑tuned exactly on the zeroshot/twitter-financial-news-topic dataset, with 20 topic labels derived from the dataset’s label mapping (e.g., LABEL_0: Analyst Update, LABEL_1: Fed | Central Banks, etc.).​

Using load_dataset("zeroshot/twitter-financial-news-topic") and your tokenizer settings is therefore appropriate. You do not need to specify num_labels manually because the model’s config already has the correct number of labels.​

Tokenization function

The tokenization step:

python
tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
is fine for quick testing, though for efficiency you might consider dynamic padding (padding=True) when you move beyond experiments.​

Evaluation and training behavior
Using train subset as eval dataset

You are currently evaluating on small_train_dataset instead of a held‑out split:

python
eval_dataset=small_train_dataset,
For a more realistic check, use the test or validation split if available, e.g.:

python
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))
...
eval_dataset=small_eval_dataset,
This will give an unbiased estimate of performance similar to the metrics reported on the model card (around 0.91 accuracy/F1 on this dataset).​

Metric function

The compute_metrics function using accuracy is standard and correct for a quick test:

python
logits, labels = eval_pred
predictions = np.argmax(logits, axis=-1)
return accuracy_metric.compute(predictions=predictions, references=labels)
If you want to fully replicate the model card’s metrics, you could additionally load and compute macro F1 and precision/recall with evaluate.load("f1"), etc., mirroring the model card values (accuracy and F1 around 0.91 on this dataset).​

In summary, if you (1) remove the hard‑coded token, (2) switch to eval_strategy="epoch" in newer transformers versions, and (3) optionally use a proper evaluation split, this script is a solid minimal setup for evaluating and quickly fine‑tuning the finance topic classifier on the Twitter financial news topic dataset.​

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


model = AutoModelForSequenceClassification.from_pretrained("finbert-tone-finetuned-finance-topic-classification",
num_labels=5)


tokenizer = AutoTokenizer.from_pretrained("finbert-tone-finetuned-finance-topic-classification")
That code will load the wrong model (or fail to load) and the num_labels=5 override does not match the checkpoint’s configuration.​

Correct model id and labels
The actual model id on Hugging Face is nickmuchi/finbert-tone-finetuned-finance-topic-classification, not just finbert-tone-finetuned-finance-topic-classification.​

This checkpoint is trained for 20 financial topic classes, so its config.num_labels is 20 and the classifier head is shaped accordingly.​

For inference with the pretrained head you should not change num_labels; just load the model as-is:

python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"

model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
When to use num_labels
Passing num_labels=5 is only appropriate if you intend to re‑initialize and fine‑tune a new 5‑class head, e.g. for your own 5‑label dataset.​

In that case you still must use the full model id, and be aware you are discarding the original 20‑class classification layer:

python
model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)
So: fix the model name to include nickmuchi/, and only set num_labels=5 if you explicitly want to train a new 5‑class classifier on top of the FinBERT backbone.​

from datasets import Dataset, DatasetDict, load_dataset


# Load train and test splits separately
#train_dataset = load_dataset("yelp_review_full", split="train[:5000]")
#test_dataset = load_dataset("yelp_review_full", split="test[:1000]")


train_dataset = load_dataset("zeroshot/twitter-financial-news-topic", split="train[:5000]")
test_dataset = load_dataset("zeroshot/twitter-financial-news-topic", split="test[:2000]")


# Create a DatasetDict
dataset = DatasetDict({
"train": train_dataset,
# "test": test_dataset,
})


print(f"data type = type(dataset)")
The DatasetDict construction is fine, but your last line just prints a literal string instead of the actual type of dataset.​

Fixing the print statement
You probably want to see the type and/or structure of the dataset object. Replace the last line with one of these:

python
print(f"data type = {type(dataset)}")
# or simply
print(dataset)
print(dataset) will show something like DatasetDict({ train: Dataset({ features: ['text', 'label'], num_rows: 5000 }) }) for this Twitter financial news topic dataset.​

import os
import torch
from huggingface_hub import login
# from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from transformers import AutoTokenizer


# 0. Set HF token (recommended: use real env var instead of hardcoding)
os.environ["HF_TOKEN"] = "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" # or set it in your shell and remove this line


# 1. Login using the token from HF_TOKEN
login(token=os.environ["HF_TOKEN"], new_session=False)


# 2. Load tokenizer and model
#model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/finbert-tone-finetuned-finance-topic-classification", num_labels=5)


#tokenizer = AutoTokenizer.from_pretrained("nickmuchi/finbert-tone-finetuned-finance-topic-classification")


model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


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


logits = model(**inputs).logits
pred_id = int(torch.argmax(logits, dim=-1))
pred_label = TOPIC_ID2LABEL.get(pred_id, f"Unknown ({pred_id})")
print(f"Predicted topic: {pred_label} (id={pred_id})")
You’re almost there; you just need to create inputs by tokenizing some text before calling the model.

Add a text string and the tokenizer call right before logits = ...:

python
import os
import torch
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 0. Set HF token (recommended: use real env var instead of hardcoding)
os.environ["HF_TOKEN"] = "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"  # remove in production

# 1. Login using the token from HF_TOKEN
login(token=os.environ["HF_TOKEN"], new_session=False)

# 2. Load tokenizer and model
model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

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

# --- NEW: build `inputs` from some text ---
text = "Fed raises interest rates amid inflation concerns."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128,
)

# Forward + decode
with torch.no_grad():
    logits = model(**inputs).logits

pred_id = int(torch.argmax(logits, dim=-1))
pred_label = TOPIC_ID2LABEL.get(pred_id, f"Unknown ({pred_id})")
print(f"Text: {text}")
print(f"Predicted topic: {pred_label} (id={pred_id})")
#Step 0: Installation of Packages
"""

#!pip uninstall -y wandb # avoid experiment tracking

#!pip install transformers[torch] -q
#!pip install dataset -q
#!pip install evaluate -q
#!pip install -U transformers

"""# load model"""
import streamlit as st
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="yiyanghkust/finbert-tone")

# Load model directly
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", dtype="auto")

"""#Step 1: Obtain your own dataset - zeroshot/twitter-financial-news-topic"""

from datasets import Dataset, DatasetDict, load_dataset

# Load train and test splits separately

train_dataset = load_dataset("zeroshot/twitter-financial-news-topic", split="train[:5000]")
val_dataset = load_dataset("zeroshot/twitter-financial-news-topic", split="validation[:1000]")

# Create a DatasetDict
dataset = DatasetDict({
"train": train_dataset,
"validation": val_dataset,
})

print(f"data type = type(dataset)")

"""# start from 5000 - 10000"""

from datasets import Dataset, DatasetDict, load_dataset

# Load train and test splits separately
train_dataset = load_dataset("zeroshot/twitter-financial-news-topic", split="train[5000:10000]")
val_dataset = load_dataset("zeroshot/twitter-financial-news-topic", split="validation[1000:2000]")
# Create a DatasetDict
dataset = DatasetDict({
"train": train_dataset,
"validation": val_dataset,
})

print(f"data type = type(dataset)")

type(dataset)

# Dataset structure
dataset

"""#Step 2: Create the model and tokenizer objects"""

import os
import torch
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 0. Set HF token (recommended: use real env var instead of hardcoding)
os.environ["HF_TOKEN"] = "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" # remove in production

# 1. Login using the token from HF_TOKEN
login(token=os.environ["HF_TOKEN"], new_session=False)

# 2. Load tokenizer and model
model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

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

# --- NEW: build `inputs` from some text ---
text = "Fed raises interest rates amid inflation concerns."

inputs = tokenizer(
text,
return_tensors="pt",
truncation=True,
padding=True,
max_length=128,
)

# Forward + decode
with torch.no_grad():
logits = model(**inputs).logits

pred_id = int(torch.argmax(logits, dim=-1))
pred_label = TOPIC_ID2LABEL.get(pred_id, f"Unknown ({pred_id})")
print(f"Text: {text}")
print(f"Predicted topic: {pred_label} (id={pred_id})")

"""#Step 3: Generate Dataset for Funetuning"""

type(dataset)

from transformers import AutoTokenizer

def tokenize_function(examples):
return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

"""# Step 4a: Finetune the pre-trained model"""

from datasets import DatasetDict

# 1. Load raw data
raw_datasets = load_dataset("zeroshot/twitter-financial-news-topic")

# 2. Tokenize with fixed length
def tokenize_function(examples):
return tokenizer(
examples["text"],
truncation=True,
padding="max_length", # ensures same length
max_length=128,
)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 3. Rename and drop columns so only model inputs remain
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# 4. Set PyTorch format with only fixed-size tensors
tokenized_datasets.set_format(
type="torch",
columns=["input_ids", "attention_mask", "labels"],
)

# 5. Subsample
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))

training_args = TrainingArguments(
output_dir="test_trainer",
num_train_epochs=1,
eval_strategy="epoch",
)

trainer = Trainer(
model=model,
args=training_args,
train_dataset=small_train_dataset,
eval_dataset=small_eval_dataset,
compute_metrics=compute_metrics,
)

trainer.train()

"""#Step 4b: Evaluate the finetuned model



"""

trainer.evaluate()

"""#Step 4c: Save the finetuned model"""

trainer.save_model('CustomModel_finbert_finance_topic_classification')

!zip -r /content/CustomModel_finbert_finance_topic_classification.zip CustomModel_finbert_finance_topic_classification

from google.colab import files
files.download("CustomModel_finbert_finance_topic_classification.zip")

"""# upload to Hugging Face"""

!hf auth login

repo_name = "korgtmf/FOMO"

model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
You now have the topic mapping and single‑text prediction wired correctly; next fixes are mainly cleanup and missing definitions so the training section runs end‑to‑end.

1. Define compute_metrics and missing imports
Add these near the top (after other imports) so Trainer works:

python
from transformers import TrainingArguments, Trainer
from evaluate import load
import numpy as np

metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
Without this, Trainer will fail because compute_metrics is undefined.

2. Remove or adapt Colab-only commands for deployment
If this script is for Streamlit deployment or running as plain Python, comment out or remove:

python
#!pip uninstall -y wandb
#!pip install ...
!zip -r /content/CustomModel_finbert_finance_topic_classification.zip ...
from google.colab import files
files.download("CustomModel_finbert_finance_topic_classification.zip")
!hf auth login
These should live in a notebook, not in a production script.​

3. Avoid hard‑coding HF token
Replace:

python
os.environ["HF_TOKEN"] = "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"
login(token=os.environ["HF_TOKEN"], new_session=False)
with something like:

python
login(token=os.environ["HF_TOKEN"], new_session=False)
and set HF_TOKEN outside the code (environment variable or Streamlit secret).​

4. (Optional) Simplify duplicate dataset blocks
You fetch two different slices into dataset:

First 0–5000

Then 5000–10000

Only the second survives. If you only want 5000–10000, delete the first block to reduce confusion.

Once those changes are in place, the topic‑id→label mapping and prediction snippet you added will work and the training pipeline will run cleanly on the zeroshot/twitter-financial-news-topic dataset with the 20‑topic FinBERT model.​

add topic id >> label mapping to below code

the import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
model_id = "korgtmf/FOMO" # your fine-tuned model on HF
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
Add the mapping once near the top, then decode pred_id to a label.

python
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

    pred_label = TOPIC_ID2LABEL.get(pred_id, f"Unknown ({pred_id})")
    st.write(f"Predicted topic: {pred_label} (id={pred_id})")
    st.bar_chart(probs.numpy())



