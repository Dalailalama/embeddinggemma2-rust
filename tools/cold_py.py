import sys, torch
from sentence_transformers import SentenceTransformer
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
m = SentenceTransformer(sys.argv[1], model_kwargs={"torch_dtype": torch.float32}, device=device)
print(len(m.encode("task: search result | query: which planet is known as the red planet")))
