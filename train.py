from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from os import remove

model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
model = AutoModelForCausalLM.from_pretrained('./ruGPT3')
tokenizer = AutoTokenizer.from_pretrained(model_name)
block_size = 64
train_dataset = TextDataset(tokenizer=tokenizer,file_path="dataset.txt",block_size=block_size)
remove(f'cached_lm_GPT2TokenizerFast_{block_size}_dataset.txt')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./finetuned", # The output directory
    overwrite_output_dir=True, # Overwrite the content of the output dir
    num_train_epochs=10, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    warmup_steps=10, # number of warmup steps for learning rate scheduler
    gradient_accumulation_steps=1, # to make "virtual" batch size larger
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    optimizers = (torch.optim.AdamW(model.parameters(),lr=1e-5), None)
)

trainer.train()
model.save_pretrained('./ruGPT3')