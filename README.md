# Book Recommendation System with Sentence Transformers (all-MiniLM-L6-v2)

## 📌 Overview

This repository hosts the quantized version of the all-MiniLM-L6-v2 model fine-tuned for book reccommendation tasks. The model has been trained on the Book_Genre dataset from Hugging Face. The model is quantized to Float16 (FP16) to optimize inference speed and efficiency while maintaining high performance.

## 🏗 Model Details

- **Model Architecture:** all-MiniLM-L6-v2
- **Task:** Book Recommendation System  
- **Dataset:** Hugging Face's `Book_Genre`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from sentence_transformers import SentenceTransformer, models
from huggingface_hub import hf_hub_download
from sentence_transformers.util import cos_sim
import torch
import ast

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/all-MiniLM-L6-v2-book-recommendation-system"
model = SentenceTransformer(model_name).to(device)
```

### Question Answer Example

```python
# Load the Embeddings CSV file
df_embeddings = pd.read_csv(csv_path)

# Convert embedding column (string) back to list
df_embeddings["embedding"] = df_embeddings["embedding"].apply(ast.literal_eval)

# Convert embeddings to tensor
book_embeddings = torch.tensor(df_embeddings["embedding"].tolist())

def get_book_recommendations(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)

    if isinstance(book_embeddings, np.ndarray):
        book_embeddings_tensor = torch.tensor(book_embeddings, dtype=torch.float32)
    else:
        book_embeddings_tensor = book_embeddings  # If already tensor, use as is

    similarities = cos_sim(query_embedding, book_embeddings_tensor)

    similarities = similarities.squeeze(0)  # Remove extra dimension if necessary

    top_k_values, top_k_indices = torch.topk(similarities, k=top_k)

    recommended_titles = df_embeddings.iloc[top_k_indices.cpu().numpy()]['title'].tolist()
    recommended_scores = top_k_values.cpu().numpy().tolist()

    return list(zip(recommended_titles, recommended_scores))

# Example usage:
query = "A horror novel with ghosts and dark nights"
recommendations = get_book_recommendations(query, top_k=5)

print("Book Recommendations:")
for title, score in recommendations:
    print(f"{title} - Score: {score:.4f}")
```

## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## Evaluation Metrics: NDCG

NDCG → If close to 1, the ranking matches expected relevance. Our model's NDCG score is 0.82

## 🔧 Fine-Tuning Details

### Dataset
The **Bookcorpus** dataset was used for training and evaluation. The dataset consists of **texts**.

### Training Configuration
- **Number of epochs**: 3
- **warmup steps**: 100
- **Evaluation strategy**: steps


## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle for out of scope tasks.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different writing styles and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
