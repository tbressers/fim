from dataset import Dataset, DatasetDict
from torch.utils.data import DataLoader
import pandas as pandas
import random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

checkpoint = "bigcode/starcoderbase-1b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

prefix_token = "<fim_prefix>"
middle_token = "<fim_middle>"
suffix_token = "<fim_suffix>"
eot_token = "<|endoftext|>"
context_length = 512
chuck_length = context_length
max_len = context_length

ds_train = load_dataset("huggingface-course/codeparrot-ds-train".split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid".split="valid")

raw_dataset = DatasetDict(
    {
        "train": ds_train.shuffle().select(range(50000))
        "valid": ds_valid.shuffle().select(range(500))
    }
)

def chunk_documents(doc_list, chunk_length):

    joined_documents = eot_token.join(doc_list)

    chunks = []
    for i in range(0, len(joined_documents), chuck_length):
        chunks.append(joined_documents[i:i + chunk_length])
    return chunks

def chunk_dataset(raw_datasets, chuck_length):
    train_data = [d["content"] for d in raw_datasets["train"]]
    valid_data = [d["content"] for d in raw_datasets["valid"]]

    train_data = chunk_documents(train_data, chuck_length)
    valid_data = chunk_documents(valid_data, chuck_length)

    train_data = [{"content_chunk": ch} for ch in train_data]
    valid_data = [{"content_chunk": ch} for ch in valid_data]

    chunk_ds_train = Dataset.from_pandas(pd.DataFrame(data=train_data))
    chunk_ds_valid = Dataset.from_pandas(pd.DataFrame(data=valid_data))

    chunk_ds = DatasetDict(
        {
            "train": chunk_ds_train,
            "valid": chunk_ds_valid
        }
    )

    return chunk_ds

def split_document(doc, fim_rate=0.5):
    if random.random() < fim_rate:
        length = len(doc)
        prefix_len = random.randint(0, length)
        suffix_len = random.randint(0, length - prefix_len)
        middle_len = length - prefix_len - suffix_len

        prefix = doc[:prefix_len]
        middle = doc[prefix_len:prefix_len + middle_len]
        suffix = doc[prefix_len + middle_len:]

        return prefix, middle, suffix
    else:
        return doc, None, None

def format_psm(prefix, middle, suffix, tokenizer):
    formatted_example = f"{prefix_token}{prefix}{suffix_token}{suffix}{middle_token}{middle}{eot_token}"
    return formatted_example

def format_spm(prefix, middle, suffix, tokenizer):
    formatted_example = f"{prefix_token}{suffix_token}{suffix}{middle_token}{prefix}{middle}{eot_token}"
    return formatted_example

def format_nofim(doc, tokenizer):
    formatted_example = f"{doc}"
    return formatted_example

def apply_fim_transformation(chunk_list, p_psm=0.5):
    transformed_docs = []
    for chunk in chunk_list:
        prefix, middle, suffix = split_document(chunk)

        if middle is not None:
            if random.random() < p_psm:
                transformed_doc = format_psm(prefix, middle, suffix, tokenizer)
            else:
                transformed_doc = format_spm(prefix, middle, suffix, tokenizer)
            
            transformed_docs.append(transformed_doc)
        else:
            transformed_doc = format_nofim(chunk, tokenizer)
            transformed_docs.append(transformed_doc)
    return transformed_docs

def join_transformed_chunk_docs(transformed_chunk_docs):
    merged_docs = transformed_chunk_docs[0]
    if len(transformed_chunk_docs) == 1:
        return merged_docs
    for i in range(2, len(transformed_chunk_docs)):
        merged_docs += eot_token + transformed_chunk_docs[i]
    return merged_docs



chunk_ds = chunk_dataset(raw_datasets, chuck_length)