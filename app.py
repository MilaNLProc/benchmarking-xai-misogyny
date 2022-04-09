import streamlit as st

SOC_DIR = "./contextualizing-hate-speech-models-with-explanations/"
import sys
if SOC_DIR not in sys.path:
    sys.path.append(SOC_DIR)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import load_metric, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from einops import rearrange
from helper import VizHelper
import matplotlib.pyplot as plt

from custom_bert import BertForSequenceClassification
import seaborn as sns; sns.set_theme()
palette = sns.diverging_palette(240, 10, as_cmap=True)

max_seq_length = 128

import logging
st.set_page_config(layout="wide")


names = {
    "Italian": {
        "model_name": "./bert-base-italian-cased_ami20/",
        "tokenizer_name": "dbmdz/bert-base-italian-cased",
        "splits": dict(
            train = pd.read_csv("data/AMI2020_training_raw_90.csv"),
            validation = pd.read_csv("data/AMI2020_validation_raw_10.csv"),
            test = pd.read_csv("data/AMI2020_test_raw_gt.tsv", sep="\t")
        ),
        "soc_kwargs": {
            "lm_dir": "./soc_lm_ami20",
            "data_dir": "./data/",
            "train_file": "AMI2020_training_raw_90.tsv",
            "valid_file": "AMI2020_validation_raw_10.tsv"
        }
    },
    "English": {
        "model_name": "./bert-base-cased_ami18/",
        "tokenizer_name": "bert-base-cased",
        "splits": dict(
            train = pd.read_csv("data/miso_train.tsv", sep="\t"),
            validation = pd.read_csv("data/miso_dev.tsv", sep="\t"),
            test = pd.read_csv("data/miso_test.tsv", sep="\t")
        ),
        "soc_kwargs": {
            "lm_dir": "./soc_lm_ami18",
            "data_dir": "./data/",
            "train_file": "miso_train.tsv",
            "valid_file": "miso_test.tsv"
        }
    }
}


@st.cache
def get_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    #effective_model = BertForSequenceClassification.from_pretrained(model_name).eval()
    return model


def generate_table(language, text):
    model = get_model(names[language]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(names[language]["tokenizer_name"])

    def preprocess_text(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_seq_length)

    train = names[language]["splits"]["train"]
    validation = names[language]["splits"]["validation"]
    test = names[language]["splits"]["test"]
    raw_datasets = DatasetDict(
        train=Dataset.from_pandas(train),
        validation=Dataset.from_pandas(validation),
        test=Dataset.from_pandas(test)
    )
    raw_datasets = raw_datasets.rename_column("misogynous", "label")
    proc_datasets = raw_datasets.map(preprocess_text, batched=True, remove_columns=raw_datasets["train"].features)
    proc_datasets.set_format("pt")

    exp = VizHelper(model, tokenizer, raw_datasets["test"], proc_datasets["test"])

    soc_kwargs = names[language]["soc_kwargs"]
    table = exp.compute_table(text, target=1, soc_kwargs=soc_kwargs)

    scores, prediction = exp.classify(text)
    return table, scores, prediction

st.title("Benchmarking XAI")

language = st.radio("Language", ["Italian", "English"])
text = st.text_input(
    "Insert your text", "Luca e' proprio un pezzo di pane"
)

if st.button("Explain") and text is not None and text != "":
    with st.spinner("Processing text..."):
        table, scores, prediction = generate_table(language, text)

        st.text(f"Prediction: {prediction}")
        st.text(f"Scores: {scores}")
        st.dataframe(table.style.background_gradient(axis=1, cmap=palette, vmin=-1, vmax=1))