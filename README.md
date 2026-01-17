# SigLIP Product Search

Fine-tuned **SigLIP** model for **e-commerce product image similarity search**, visual recommendation, and retrieval.

- Base model: `google/siglip-base-patch16-224`
- Fine-tuned on: Amazon Berkeley Objects (**ABO**) small dataset
- Embedding size: **512** dimensions (L2-normalized)
- Focus: fast CPU inference + high-quality product similarity

## ✨ Features

- Zero-shot and fine-tuned product-to-product similarity
- Produces semantically rich 512-dim embeddings
- Works well for fashion, home goods, electronics, etc.
- Lightweight → runs efficiently even on CPU

## Installation

```bash
git clone https://github.com/ishtiyaqe/siglip-product-search.git
cd siglip-product-search

# Recommended: use a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Requirements (requirements.txt summary)

Typical dependencies include:

- `torch`
- `transformers` (>=4.40 recommended)
- `pillow`
- `numpy`
- `tqdm`
- `scikit-learn` or `faiss-cpu` (for nearest neighbor search — optional)

## Quick Start

### 1. Get embeddings from images

```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

# Load fine-tuned model & processor
model = AutoModel.from_pretrained("ishtiyaqe/siglip-product-search")   # ← change to local path if needed
processor = AutoProcessor.from_pretrained("ishtiyaqe/siglip-product-search")

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Example image
img = Image.open("product.jpg")

inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    image_embeds = model.get_image_features(**inputs)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)  # L2 normalize

print(image_embeds.shape)          # torch.Size([1, 512])
print(image_embeds[0, :8])         # first values of normalized embedding
```

### 2. Simple similarity search (example)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Suppose you have many product embeddings in a matrix `gallery_embeds` (n_products × 512)
# and query embedding `query_embed` (1 × 512)

similarities = cosine_similarity(query_embed.cpu().numpy(), gallery_embeds)[0]
top_k_indices = np.argsort(similarities)[::-1][:8]

print("Top similar products indices:", top_k_indices)
print("Similarity scores:", similarities[top_k_indices])
```

For large-scale search → use **FAISS**, **Annoy**, or **usearch**.

## Model Download / Usage

You can also load the model directly from Hugging Face (if you upload it there):

```python
model = AutoModel.from_pretrained("ishtiyaqe/siglip-product-search")
processor = AutoProcessor.from_pretrained("ishtiyaqe/siglip-product-search")
```

(Upload command – from repo root:)

```bash
huggingface-cli login
huggingface-cli upload ishtiyaqe/siglip-product-search . --repo-type model
```

## Performance Notes

- Inference time (224×224 image, CPU): ~80–200 ms depending on hardware
- Much better product understanding than original SigLIP on e-commerce images
- Still multilingual-capable (thanks to SigLIP base)



## Todo / Roadmap

- [ ] Upload fine-tuned weights to Hugging Face Hub
- [ ] Add FAISS index building & search example
- [ ] Gradio / Streamlit demo app

## Acknowledgments

- Google Research — [SigLIP paper](https://arxiv.org/abs/2303.15343)
- Amazon Berkeley Objects Dataset
- Hugging Face Transformers team

Made with ❤️ 

