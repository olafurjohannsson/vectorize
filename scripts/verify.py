from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texts = ["Hello world", "How are you?"]

encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Input IDs: {encoded['input_ids'].tolist()}")
print(f"Attention mask: {encoded['attention_mask'].tolist()}")

with torch.no_grad():
    outputs = model(**encoded)

    hidden_states = outputs.last_hidden_state
    print(f"Hidden states shape: {hidden_states.shape}")

    attention_mask = encoded['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"First 5 values of first embedding: {embeddings[0][:5].tolist()}")
    print(f"First embedding norm: {torch.norm(embeddings[0]).item()}")

    print(f"\nFull first 10 values:")
    print(embeddings[0][:10].tolist())

    print(f"\nStatistics for first embedding:")
    print(f"Mean: {embeddings[0].mean().item()}")
    print(f"Std: {embeddings[0].std().item()}")
    print(f"Min: {embeddings[0].min().item()}")
    print(f"Max: {embeddings[0].max().item()}")