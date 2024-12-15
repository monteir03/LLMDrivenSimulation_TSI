from transformers import AutoTokenizer, AutoModelForCausalLM

# Path to your fine-tuned model directory
fine_tuned_model_dir = "./observation-qwen-finetuned"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir)
model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_dir).to("mps")  # Use "mps" or "cpu" for M1

print("Model and tokenizer loaded successfully!")


# Example input for testing
input_text = """You are the Observation Agent in a multi-agent system. Your task is to analyze the state of the container and determine which agent should be activated: the Adding Agent or the Mixing Agent.

### Your Task:
1. Check if the container is full or not:
   - **Full**: All 10 rows are filled with no empty cells (no zeroes).
   - **Not Full**: If there is at least one empty cell or fewer than 10 rows of balls.
2. If the container is not full, the correct agent to call is the **Adding Agent**.
3. If the container is full, the correct agent to call is the **Mixing Agent**.

### Input:
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [2 2 2 2 2 2 2 2 2 2]
 [2 2 2 2 2 2 2 2 2 2]
 [2 2 2 2 2 2 2 2 2 2]
 [1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]]

### Output:
"""

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("mps")  # Use "mps" or "cpu" for M1


# Generate predictions
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=512,  # Set an appropriate max length for output
    num_beams=4,  # Use beam search for better quality
    no_repeat_ngram_size=2,  # Prevent repetition
    early_stopping=True
)

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Output:")
print(generated_text)
