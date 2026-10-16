"""Reference text embeddings from the original PyTorch / sentence-transformers implementation (float32)."""
import json, sys, torch
from sentence_transformers import SentenceTransformer

MODEL = sys.argv[1]
OUT = sys.argv[2]

model = SentenceTransformer(MODEL, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
print("loaded", type(model).__name__, file=sys.stderr)

LONG = " ".join(f"Sentence number {i} talks about topic {i % 17} in some detail." for i in range(150))
cases = [
    ("SearchQuery", "which planet is known as the red planet"),
    ("Document", "Mars, known as the red planet, has two small moons named Phobos and Deimos."),
    ("SearchQuery", "日本の首都はどこですか？"),
    ("Document", "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"),
    (None, "x"),
    ("Document", LONG),  # > 1000 tokens: exercises the sliding-window mask and global layers
]
res = []
for prompt, text in cases:
    kw = {"prompt_name": prompt} if prompt else {}
    feats = model.tokenize([text]) if False else None
    emb = model.encode(text, convert_to_numpy=True, **kw)
    full = (prompt and model.prompts[prompt] or "") + text
    tok = model.tokenizer(full, return_tensors=None)["input_ids"]
    entry = {"prompt": prompt, "text": text, "ids": tok, "emb": emb.tolist()}
    for d in (512, 256, 128):
        entry[f"emb{d}"] = model.encode(text, truncate_dim=d, normalize_embeddings=True, **kw).tolist()
    res.append(entry)
    print(prompt, len(tok), "tokens, |emb|=", float((emb ** 2).sum() ** 0.5), file=sys.stderr)
json.dump(res, open(OUT, "w"))
