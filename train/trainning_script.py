from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

# 1. Load the dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "train_observation_agent_dataset.jsonl",
        "validation": "validation_observation_agent_dataset.jsonl"
    }
)

# 2. Load the tokenizer
model_name = "Qwen/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# 3. Preprocessing Function
def preprocess_function(examples):
    input_texts = []
    output_texts = []

    for input_matrix, output_text in zip(examples['input'], examples['output']):
        input_text = f"""You are the Observation Agent in a multi-agent system. Your task is to analyze the state of the container and determine which agent should be activated: the Adding Agent or the Mixing Agent.

### Your Task:
1. Check if the container is full or not:
   - **Full**: All 10 rows are filled with no empty cells (no zeroes).
   - **Not Full**: If there is at least one empty cell or fewer than 10 rows of balls.
2. If the container is not full, the correct agent to call is the **Adding Agent**.
3. If the container is full, the correct agent to call is the **Mixing Agent**.

### Input:
{input_matrix}

### Output:
"""
        input_texts.append(input_text)
        output_texts.append(output_text)

    # Tokenize input and output texts
    tokenized_inputs = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",
        max_length=512,  # Adjust max length to fit memory
    )

    tokenized_outputs = tokenizer(
        output_texts,
        truncation=True,
        padding="max_length",
        max_length=512,  # Adjust max length to fit memory
    )

    # Add labels (same as tokenized outputs)
    tokenized_inputs["labels"] = tokenized_outputs["input_ids"]

    return tokenized_inputs


# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,  # Enables remote model code
).to("mps")  # Use Metal Performance Shaders (MPS)


print("model:", model)

# 5. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For causal language modeling
    r=4,  # Rank of low-rank adapters
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target modules
    lora_dropout=0.1,  # Dropout for regularization
    bias="none"  # No bias fine-tuning
)

# Apply LoRA configuration to the model
model = get_peft_model(model, lora_config)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./observation-qwen-finetuned",  # Directory to save the model
    evaluation_strategy="steps",  # Match save strategy
    save_strategy="steps",  # Save strategy to match evaluation strategy
    save_steps=50,  # Save every 50 steps
    eval_steps=50,  # Evaluate every 50 steps to align with save strategy
    learning_rate=1e-4,  # Learning rate
    per_device_train_batch_size=1,  # Batch size for low memory
    gradient_accumulation_steps=8,  # Accumulate gradients
    num_train_epochs=1,  # Reduce epochs for MPS testing
    logging_steps=10,  # Log every 10 steps
    save_total_limit=1,  # Keep only the latest checkpoint
    fp16=False,  # FP16 is not supported on M1
    bf16=False,  # BF16 is also not supported
    logging_dir="./logs",  # Log directory
    report_to="none",  # Disable external logging
    load_best_model_at_end=True  # Automatically load the best model at the end
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# 8. Train the Model
trainer.train()

# 9. Save the Model
model.save_pretrained("./observation-qwen-finetuned")
tokenizer.save_pretrained("./observation-qwen-finetuned")
