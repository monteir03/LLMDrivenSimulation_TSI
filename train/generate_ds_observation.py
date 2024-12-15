import json
import random

def generate_matrix(fill_probability=0.9):
    """Generate a 10x10 matrix with random ball distribution under gravity."""
    matrix = [[0] * 10 for _ in range(10)]  # Start with an empty matrix (all zeros)
    filled_rows = 0

    for _ in range(10):
        if random.random() < fill_probability and filled_rows < 10:
            # Fill the next row at the bottom-most empty position
            ball_type = random.choice([1, 2, 3])  # Choose a single ball type
            matrix[9 - filled_rows] = [ball_type] * 10  # Fill the row
            filled_rows += 1
        else:
            # Stop filling if the row is skipped due to probability or matrix is full
            break

    return matrix

def process_matrix_to_string(matrix):
    matrix_string = "["
    for row in matrix:
        row_string = " [" + " ".join(map(str, row)) + "]"
        matrix_string += row_string + "\n"
    matrix_string += "]"
    return matrix_string


def is_full(matrix):
    """Check if the matrix is fully filled."""
    return all(cell != 0 for row in matrix for cell in row)

def label_matrix(matrix):
    """Label the matrix based on its fullness."""
    return {
        "is_full": is_full(matrix),
        "agent_to_call": "Mixing Agent" if is_full(matrix) else "Adding Agent"
    }

def process_label_to_string(matrix):
    """
    Label the matrix and return a string representation of the output in JSON format.
    """
    label = {
        "is_full": is_full(matrix),
        "agent_to_call": "Mixing Agent" if is_full(matrix) else "Adding Agent"
    }

    # Convert the label to a JSON string without escaping quotes
    return str(label)  # Escapes quotes for embedding


def generate_dataset(num_samples=100):
    """Generate a labeled dataset."""
    dataset = []
    for _ in range(num_samples):
        numeric_matrix = generate_matrix()
        label = process_label_to_string(numeric_matrix)
        matrix = process_matrix_to_string(numeric_matrix)
        dataset.append({"input": matrix, "output": label})
    return dataset

# Generate and save the dataset
dataset_train = generate_dataset(20)
with open("train_observation_agent_dataset.jsonl", "w") as f:
    for entry in dataset_train:
        f.write(json.dumps(entry) + "\n")

validation_dataset = generate_dataset(5)
with open("validation_observation_agent_dataset.jsonl", "w") as f:
    for entry in generate_dataset(100):
        f.write(json.dumps(entry) + "\n")