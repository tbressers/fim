from dataset import Dataset, DatasetDict
from torch.utils.data import DataLoader
import pandas as pandas
import random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling


device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("finetuned_model")
model = AutoModelForCausalLM.from_pretrained("finetuned_model").to(device)

input_text = "<fim_prefix>def print_hello..."

inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
