# Creating the `llama3_zero:latest` Model with Ollama

This guide explains how to use Ollama to create a customized version of the `llama3` model, specifically named **`llama3_zero:latest`**, with **zero temperature**.

---

## Prerequisites

Make sure you have the following:

- [Ollama CLI](https://ollama.ai/) installed.
- A pre-existing `Modelfile` that defines the customization.

---

## Steps to Create the Custom Model

### 1. Create the Model

Run the following command in the directory where your `Modelfile` is located:

```bash
ollama create llama3_zero -f ./Modelfile
```

- `llama3_zero`: The name for the new model.
- `./Modelfile`: Path to the pre-existing model configuration file.

The model will be tagged as `latest` automatically.

---

### 2. Verify the Model

To confirm the model has been created successfully, list all models:

```bash
ollama list
```

You should see the new model `llama3_zero:latest` in the list.

---

### 3. Run the Custom Model

To test or use your newly created model, execute:

```bash
ollama run llama3_zero
```

This will load the model and allow you to interact with it.
