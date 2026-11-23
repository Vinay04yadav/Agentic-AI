"""
Vector embeddings demo: what they look like, how similarity is computed,
and why similar words cluster together.

Install:  pip install sentence-transformers scikit-learn matplotlib numpy faiss-cpu
Run:      python embeddings_demo.py
          python embeddings_demo.py ocean        # also rank all words against "ocean"
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------- data
GROUPS = {
    "animals": ["cat", "dog", "lion", "puppy"],
    "royalty": ["king", "queen", "prince", "throne"],
    "food":    ["pizza", "burger", "pasta", "banana"],
    "tech":    ["computer", "laptop", "python", "algorithm"],
}
COLORS = {"animals": "tab:orange", "royalty": "tab:purple",
          "food": "tab:green", "tech": "tab:blue"}

words, labels = [], []
for g, ws in GROUPS.items():
    words += ws
    labels += [g] * len(ws)
point_colors = [COLORS[g] for g in labels]

# ----------------------------------------------------------------- embed
model = SentenceTransformer("all-MiniLM-L6-v2")      # 384-dim vectors
emb = model.encode(words)                            # shape (n_words, 384)
print(f"Embedding matrix shape: {emb.shape}  (words x dimensions)")


# ----------------------------------------------------------------- similarity
def cosine_breakdown(a, b):
    """Return dot product, norms, cosine, and angle: shows the math step by step."""
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cos = dot / (na * nb)
    angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return dot, na, nb, cos, angle


norms = np.linalg.norm(emb, axis=1, keepdims=True)
unit = emb / norms
sim = unit @ unit.T                                  # full cosine similarity matrix

# Optional: rank every word against a query given on the command line
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
    q = model.encode([query])[0]
    q = q / np.linalg.norm(q)
    scores = unit @ q
    print(f'\nSimilarity of every word to "{query}":')
    for i in np.argsort(-scores):
        print(f"  {words[i]:<10} {scores[i]:+.3f}  {'#' * int(max(scores[i], 0) * 40)}")

# ----------------------------------------------------------------- numeric view + storage
import os
import sqlite3

np.set_printoptions(precision=4, suppress=True, linewidth=110)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# 1) the actual numbers
section("1) THE ACTUAL NUMBERS: embedding of 'cat'")
v = emb[words.index("cat")]
print(f"type={type(v).__name__}  dtype={v.dtype}  shape={v.shape}  "
      f"bytes={v.nbytes}  ({v.size} dims x {v.itemsize} bytes)")
print("first 12 values:", v[:12])
print("last 6 values  :", v[-6:])
print(f"min={v.min():+.4f}  max={v.max():+.4f}  mean={v.mean():+.4f}  L2 norm={np.linalg.norm(v):.4f}")
print("\nSame 8 dimensions for different words:")
for w in ["cat", "dog", "pizza"]:
    print(f"  {w:<6}", emb[words.index(w), :8])

# 2) raw bytes
section("2) HOW THOSE NUMBERS SIT IN MEMORY / ON DISK (raw bytes)")
raw = v.astype(np.float32).tobytes()
print(f"{len(raw)} bytes in total. The first 3 floats become:")
for i in range(3):
    print(f"  dim {i}: {v[i]:+.6f}  ->  bytes {raw[i*4:(i+1)*4].hex(' ')}   (float32, 4 bytes, little-endian)")
back = np.frombuffer(raw, dtype=np.float32)
print("bytes -> numpy round trip identical:", np.array_equal(back, v))
np.save("embeddings.npy", emb.astype(np.float32))
print(f"saved the whole matrix to embeddings.npy ({os.path.getsize('embeddings.npy')} bytes: "
      f"{emb.shape[0]}x{emb.shape[1]}x4 + small header)")

# 3) a tiny vector 'database' in SQLite
section("3) A TINY VECTOR STORE (SQLite: vector lives in a BLOB column next to metadata)")
if os.path.exists("embeddings_demo.db"):
    os.remove("embeddings_demo.db")
con = sqlite3.connect("embeddings_demo.db")
con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, word TEXT, category TEXT, vector BLOB)")
con.executemany(
    "INSERT INTO items VALUES (?,?,?,?)",
    [(i, w, g, emb[i].astype(np.float32).tobytes()) for i, (w, g) in enumerate(zip(words, labels))],
)
con.commit()
print(f"{'id':<3} {'word':<10} {'category':<9} vector (BLOB)")
for rid, w, g, blob in con.execute("SELECT id, word, category, vector FROM items LIMIT 5"):
    print(f"{rid:<3} {w:<10} {g:<9} <{len(blob)} bytes> {blob[:12].hex(' ')} ...")
print("(open embeddings_demo.db in 'DB Browser for SQLite' to inspect it yourself)")

rows = con.execute("SELECT word, vector FROM items").fetchall()
mat = np.stack([np.frombuffer(b, dtype=np.float32) for _, b in rows])
qv = mat[words.index("cat")]
exact = (mat @ qv) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv))
top = np.argsort(-exact)[:4]
print("\nBrute-force search loaded from the DB, nearest to 'cat':",
      [(rows[i][0], round(float(exact[i]), 3)) for i in top])
print(f"That compared against all {len(rows)} vectors. At 10M vectors this is too slow -> ANN index.")

# 4) a real index (FAISS)
section("4) A REAL VECTOR INDEX (FAISS): what vector DBs build on top of the raw vectors")
try:
    import faiss

    dim = emb.shape[1]
    xb = np.ascontiguousarray(unit, dtype=np.float32)  # unit length -> inner product == cosine
    flat = faiss.IndexFlatIP(dim)
    flat.add(xb)
    print(f"IndexFlatIP  : ntotal={flat.ntotal}, dim={flat.d}. Raw vectors, exact search.")
    hnsw = faiss.IndexHNSWFlat(dim, 8, faiss.METRIC_INNER_PRODUCT)
    hnsw.add(xb)
    print(f"IndexHNSWFlat: ntotal={hnsw.ntotal}. Raw vectors + a graph linking near neighbours "
          f"(each node keeps ~8 links), approximate search.")
    q1 = xb[words.index("cat")][None, :]
    for name, index in [("Flat", flat), ("HNSW", hnsw)]:
        D, I = index.search(q1, 4)
        print(f"  {name:<5} top-4 for 'cat':", [(words[i], round(float(s), 3)) for i, s in zip(I[0], D[0])])
    print("  vector id 0 reconstructed from index matches original:", np.allclose(flat.reconstruct(0), xb[0]))
    faiss.write_index(flat, "embeddings.faiss")
    print(f"  saved index to embeddings.faiss ({os.path.getsize('embeddings.faiss')} bytes)")
except ImportError:
    print("faiss not installed, skipping. Install with: pip install faiss-cpu")

# 5) storage cost and compression
section("5) STORAGE COST & COMPRESSION")
print("Size of 1,000,000 vectors (vectors only, no index or metadata):")
print(f"  {'precision':<10} {'384 dims':>12} {'1536 dims':>12}")
for name, bytes_per_dim in [("float32", 4), ("float16", 2), ("int8", 1), ("binary", 1 / 8)]:
    print(f"  {name:<10} {384 * bytes_per_dim * 1e6 / 1e9:>9.2f} GB {1536 * bytes_per_dim * 1e6 / 1e9:>9.2f} GB")

scale = 127.0 / np.abs(v).max()
v_int8 = np.round(v * scale).astype(np.int8)
v_back = v_int8.astype(np.float32) / scale
v_fp16 = v.astype(np.float16).astype(np.float32)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n'cat' vector after compression:")
print("  original float32 :", v[:6])
print("  int8 (stored)    :", v_int8[:6], f"-> cosine to original {cos(v, v_back):.5f}")
print("  float16          :", v_fp16[:6], f"-> cosine to original {cos(v, v_fp16):.5f}")
print("  => 4x smaller with int8 and similarity barely changes.")


# ----------------------------------------------------------------- Figure 1: 2x2 overview
fig, ax = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Vector embeddings: 384-dim vectors, all-MiniLM-L6-v2", fontsize=14)

# (A) similarity heatmap
a = ax[0, 0]
im = a.imshow(sim, cmap="viridis", vmin=0, vmax=1)
a.set_xticks(range(len(words)))
a.set_yticks(range(len(words)))
a.set_xticklabels(words, rotation=90, fontsize=8)
a.set_yticklabels(words, fontsize=8)
a.set_title("A) Cosine similarity matrix\n(bright squares along the diagonal = clusters)")
fig.colorbar(im, ax=a, fraction=0.046)

# (B) 2D PCA projection
b = ax[0, 1]
pca2 = PCA(n_components=2).fit(emb)
p2 = pca2.transform(emb)
b.scatter(p2[:, 0], p2[:, 1], c=point_colors, s=90)
for (x, y), w in zip(p2, words):
    b.annotate(w, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=9)
b.set_title(f"B) 384-D squashed to 2-D with PCA\n"
            f"(keeps only {pca2.explained_variance_ratio_.sum():.0%} of the variance: a lossy shadow)")
b.set_xlabel("PC1")
b.set_ylabel("PC2")
b.grid(alpha=0.3)
for g, c in COLORS.items():
    b.scatter([], [], c=c, label=g)
b.legend()

# (C) raw numbers: the first 48 dims of a few words
c = ax[1, 0]
show = ["cat", "dog", "pizza", "king", "queen"]
idx = [words.index(w) for w in show]
im2 = c.imshow(emb[idx, :48], aspect="auto", cmap="coolwarm", vmin=-0.3, vmax=0.3)
c.set_yticks(range(len(show)))
c.set_yticklabels(show)
c.set_xlabel("dimension (first 48 of 384)")
c.set_title("C) What the raw vectors look like\n"
            "(no single column is readable; similar words just have similar overall patterns)")
fig.colorbar(im2, ax=c, fraction=0.046)

# (D) the similarity math, step by step
d = ax[1, 1]
d.axis("off")
pairs = [("cat", "dog"), ("king", "queen"), ("cat", "pizza"), ("king", "banana")]
lines = ["D) How cosine similarity is computed\n",
         "cos(a,b) = (a . b) / (|a| * |b|)\n"]
for w1, w2 in pairs:
    dot, na, nb, cs, ang = cosine_breakdown(emb[words.index(w1)], emb[words.index(w2)])
    lines.append(f"{w1} vs {w2}")
    lines.append(f"  a.b = {dot:6.3f}   |a| = {na:.3f}   |b| = {nb:.3f}")
    lines.append(f"  cosine = {cs:+.3f}   angle = {ang:5.1f} deg")
    lines.append("")
lines.append("1.0 = same direction (very similar)")
lines.append("0.0 = perpendicular (unrelated)")
d.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=10)

plt.tight_layout()

# ----------------------------------------------------------------- Figure 2: 3D PCA
fig3 = plt.figure(figsize=(8, 7))
a3 = fig3.add_subplot(111, projection="3d")
pca3 = PCA(n_components=3).fit(emb)
p3 = pca3.transform(emb)
a3.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=point_colors, s=70)
for (x, y, z), w in zip(p3, words):
    a3.text(x, y, z, " " + w, fontsize=8)
a3.set_title(f"3-D PCA projection (rotate me!)\n"
             f"keeps {pca3.explained_variance_ratio_.sum():.0%} of the variance; "
             f"the real space has 384 dimensions")
a3.set_xlabel("PC1")
a3.set_ylabel("PC2")
a3.set_zlabel("PC3")

plt.show()