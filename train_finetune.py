#Step 0: Installation of Packages
"""

#!pip uninstall -y wandb     # avoid experiment tracking

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
from huggingface_hub import login
# from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers import AutoTokenizer

# 0. Set HF token (recommended: use real env var instead of hardcoding)
#os.environ["HF_TOKEN"] = "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"  # or set it in your shell and remove this line

# 1. Login using the token from HF_TOKEN
#login(token=os.environ["HF_TOKEN"], new_session=False)

# 2. Load tokenizer and model
#model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/finbert-tone-finetuned-finance-topic-classification", num_labels=5)

#tokenizer = AutoTokenizer.from_pretrained("nickmuchi/finbert-tone-finetuned-finance-topic-classification")

model_id = "nickmuchi/finbert-tone-finetuned-finance-topic-classification"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

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
        padding="max_length",  # ensures same length
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
