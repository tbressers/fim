from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
import pandas as pd
import random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

model_id = "meta-llama/Llama-3.2-3B-Instruct"

prefix_token = "<fim_prefix>"
middle_token = "<fim_middle>"
suffix_token = "<fim_suffix>"
eot_token = "<|endoftext|>"
context_length = 512
chuck_length = context_length
max_len = context_length

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train.shuffle().select(range(50000)),
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

def apply_context_level_fim(chunk):
    chunk_docs = chunk.split(eot_token)
    join_transformed_chunk_doc = apply_fim_transformation(chunk_docs)
    joined_transformed_chunk_docs = join_transformed_chunk_docs(join_transformed_chunk_doc)

    formatted_example = tokenizer(
        joined_transformed_chunk_docs,
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )

    return {
        "input_ids": torch.tensor(formatted_example["input_ids"]).unsqueeze(0),
        "attention_mask": torch.tensor(formatted_example["attention_mask"]).unsqueeze(0)
    }

class FimDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        return f"Dataset(num_rows={len(self)})"
    
    def __getitem__(self, index):
        chunk = self._data[index]["content_chunk"]
        chunk = str(chunk)
        return apply_context_level_fim(chunk)

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()
        data_collator = self.data_collator
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif not isinstance(eval_dataset, Dataset):
            raise ValueError("eval_dataset should be an instance of datasets.Dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)
        data_collator = self.data_collator
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

chunk_ds = chunk_dataset(raw_datasets, chuck_length)
train_dataset = FimDataset(chunk_ds["train"])
eval_dataset = FimDataset(chunk_ds["valid"])

# Lora configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    use_rslora=True
)

# Setup bits and bytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=bnb_config)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=f"./models/{model_id}",
    logging_dir=f"./logging/{model_id}",
    save_strategy="steps",
    num_train_epochs=16,
    save_steps=200,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=8,
    learning_rate=5e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    bf16=False,
    push_to_hub=False,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    logging_strategy="steps",
    logging_steps=1,
    warmup_ratio=0.03,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    remove_unused_columns = False,
    report_to="tensorboard"
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("finetuned_model")
