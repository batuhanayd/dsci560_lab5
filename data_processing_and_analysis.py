import sys
import io
import json
import os
import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import requests as res
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
_TESSERACT_EXE = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe" if os.name == "nt" else None
)
if _TESSERACT_EXE and os.path.isfile(_TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_EXE
try:
    from pytesseract import TesseractNotFoundError
except ImportError:
    TesseractNotFoundError = FileNotFoundError

import nltk
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec


def _log(msg):
    print(msg, flush=True)


_log("[START] Loading NLTK data (punkt)...")
nltk.download("punkt", quiet=True)
_log("[OK] NLTK ready.")


def _find_tesseract():
    """Set pytesseract.tesseract_cmd if Tesseract is installed but not in PATH."""
    if getattr(pytesseract.pytesseract, "tesseract_cmd", None):
        return
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return
    home = os.environ.get("TESSERACT_HOME", "").rstrip(os.sep)
    if home:
        exe = os.path.join(home, "tesseract.exe")
        if os.path.isfile(exe):
            pytesseract.pytesseract.tesseract_cmd = exe
            return
    if os.name == "nt":
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path
            return
        for base in (
            os.environ.get("ProgramFiles", "C:\\Program Files"),
            os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
            os.path.expandvars(r"%LocalAppData%\Programs"),
        ):
            for sub in ("Tesseract-OCR", "Tesseract"):
                exe = os.path.join(base, sub, "tesseract.exe")
                if os.path.isfile(exe):
                    pytesseract.pytesseract.tesseract_cmd = exe
                    return


def run_ocr(csv_path="reddit_posts.csv", image_dir="data/images", processed_data_dir="data/processed_data"):
    """Load image_urls from CSV; run OCR only for URLs not already in reddit_ocr_results.json; merge and save."""
    _log("[OCR] Step 1/3: Reading CSV...")
    image_urls = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("image_urls"):
                image_urls.append(row["image_urls"])

    all_urls = []
    for urls in image_urls:
        for url in urls.split("|"):
            url = url.strip()
            if url and url.startswith("http"):
                all_urls.append(url)
    all_urls = list(dict.fromkeys(all_urls))  # dedupe, keep order

    out_path = os.path.join(processed_data_dir, "reddit_ocr_results.json")
    existing_ocr = {}
    if os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_ocr = json.load(f)
            _log(f"[OCR] Loaded {len(existing_ocr)} already-extracted URL(s) from existing JSON.")
        except Exception as e:
            _log(f"[OCR] Could not load existing JSON: {e}")

    urls_to_process = [u for u in all_urls if u not in existing_ocr]
    _log(f"[OCR] Found {len(all_urls)} image URL(s) total; {len(urls_to_process)} new (will run OCR only on these).")
    if not all_urls:
        _log("[OCR] No images; writing empty OCR JSON and continuing.")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    dic = dict(existing_ocr)
    if not urls_to_process:
        _log("[OCR] No new images to process. Keeping existing OCR results.")
        return

    _find_tesseract()
    ocr_available = True
    for j, url in enumerate(urls_to_process):
        _log(f"[OCR] New image {j+1}/{len(urls_to_process)}: fetching...")
        try:
            resp = res.get(url, timeout=10)
            img = Image.open(io.BytesIO(resp.content))
        except Exception as e:
            _log(f"[OCR] Skip (fetch failed): {e}")
            dic[url] = None
            continue
        extension = ".png"
        file_name = f"image_new_{j}{extension}"
        file_path = os.path.join(image_dir, file_name)
        img.save(file_path)
        text = ""
        if ocr_available:
            try:
                _log(f"[OCR] New image {j+1}/{len(urls_to_process)}: running Tesseract...")
                text = pytesseract.image_to_string(img, lang="eng").strip()
            except (TesseractNotFoundError, FileNotFoundError):
                ocr_available = False
                _log("Tesseract not found; skipping OCR (using empty text for images).")
                _log("  Install from https://github.com/UB-Mannheim/tesseract/wiki or set TESSERACT_CMD to the full path of tesseract.exe")
        dic[url] = text if text else None

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)
    _log(f"[OCR] Done. Results saved to {out_path} (total {len(dic)} URLs).")


def string_combination(row, ocr_url):
    """
    Combine a row of raw data with the corresponding OCR text if images exist.
    """
    lines = []
    for value in row.values():
        if isinstance(value, str):
            val = value.strip()
        else:
            val = value
        if val is None:
            val = ""
        lines.append(f"{val},")

    image_urls = row.get("image_urls", "") or ""
    if image_urls.strip():
        urls = [u.strip() for u in image_urls.split("|") if u.strip().startswith("http")]
        for url in urls:
            ocr_text = ocr_url.get(url)
            if ocr_text is None:
                ocr_text = ""
            lines.append(f"{ocr_text},")

    return "\n".join(lines)


def doc_combination(csv_path, json_path):
    """Read CSV and OCR JSON; return list of document strings for embedding."""
    ocr_url = {}
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            ocr_url = json.load(f)

    doc_strings = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_strings.append(string_combination(row, ocr_url))
    return doc_strings


def doc_to_vector(model, tokens):
    """Convert a list of tokens to the mean of their Word2Vec vectors; zero vector if none in vocab."""
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def run_embedding(
    csv_path="reddit_posts.csv",
    json_path="data/processed_data/reddit_ocr_results.json",
    out_dir="data/processed_data",
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
):
    """Build document strings, train Word2Vec, compute doc vectors as mean of word vectors, save."""
    _log("[EMBED] Step 2/3: Building document strings from CSV + OCR...")
    doc_strings = doc_combination(csv_path, json_path)
    _log(f"[EMBED] Got {len(doc_strings)} documents.")
    os.makedirs(out_dir, exist_ok=True)
    vectors_path = os.path.join(out_dir, "doc_vectors.npy")
    model_path = os.path.join(out_dir, "word2vec.model")

    _log("[EMBED] Tokenizing and training Word2Vec...")
    tokenized_docs = [word_tokenize(doc.lower()) for doc in doc_strings]
    w2v_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0,
        seed=42,
    )
    _log("[EMBED] Computing doc vectors (mean of word vectors)...")
    doc_vectors = np.array([doc_to_vector(w2v_model, tokens) for tokens in tokenized_docs], dtype=np.float64)
    np.save(vectors_path, doc_vectors)
    w2v_model.save(model_path)
    _log(f"[EMBED] Done. Vectors saved to {vectors_path}, model to {model_path}")
    return None, doc_vectors


MIN_CLUSTER_SIZE = 5
MIN_SILHOUETTE = 0.02
MAX_LEAVES = 50


@dataclass
class ClusterNode:
    """Binary tree node: leaf (indices) or internal (left/right children)."""
    indices: np.ndarray
    left: Optional["ClusterNode"] = None
    right: Optional["ClusterNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def size(self) -> int:
        return len(self.indices)


def split_quality(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score for 2-way split; -1 if only one label."""
    if len(np.unique(labels)) < 2:
        return -1.0
    try:
        return float(silhouette_score(vectors, labels))
    except Exception:
        return -1.0


def build_cluster_tree(
    vectors: np.ndarray,
    indices: Optional[np.ndarray] = None,
    min_size: int = MIN_CLUSTER_SIZE,
    min_silhouette: float = MIN_SILHOUETTE,
    max_leaves: Optional[int] = MAX_LEAVES,
) -> ClusterNode:
    """Build binary cluster tree; stop by size, silhouette, or max_leaves."""
    if indices is None:
        indices = np.arange(len(vectors))

    n = len(indices)
    X = vectors[indices]

    if n < min_size:
        return ClusterNode(indices=indices)
    if max_leaves is not None and max_leaves <= 1:
        return ClusterNode(indices=indices)

    clustering = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
    labels = clustering.fit_predict(X)
    score = split_quality(X, labels)

    if score < min_silhouette:
        return ClusterNode(indices=indices)

    left_mask = labels == 0
    right_mask = labels == 1
    left_indices = indices[left_mask]
    right_indices = indices[right_mask]

    if len(left_indices) == 0 or len(right_indices) == 0:
        return ClusterNode(indices=indices)

    left_budget = max_leaves // 2 if max_leaves is not None else None
    right_budget = (max_leaves - left_budget) if max_leaves is not None else None

    node = ClusterNode(indices=indices)
    node.left = build_cluster_tree(vectors, left_indices, min_size, min_silhouette, max_leaves=left_budget)
    node.right = build_cluster_tree(vectors, right_indices, min_size, min_silhouette, max_leaves=right_budget)
    return node


def print_tree(node: ClusterNode, depth: int = 0, prefix: str = "") -> None:
    """Print tree structure (left = L, right = R)."""
    if node.is_leaf:
        print(f"{prefix}[leaf] size={node.size()}")
        return
    print(f"{prefix}[split] size={node.size()}")
    print_tree(node.left, depth + 1, prefix + "  L: ")
    print_tree(node.right, depth + 1, prefix + "  R: ")


def leaves(node: ClusterNode) -> list:
    """Return all leaf nodes (each is a cluster of doc indices)."""
    if node.is_leaf:
        return [node]
    return leaves(node.left) + leaves(node.right)


def _layout(node: ClusterNode, depth: int, x_start: float, x_end: float, pos: dict) -> float:
    """Assign (x,y) to each node; x spans [x_start, x_end] for subtree. Returns next x."""
    id_ = id(node)
    if node.is_leaf:
        x = (x_start + x_end) / 2
        pos[id_] = (x, -depth)
        return (x_start + x_end) / 2
    mid = (x_start + x_end) / 2
    _layout(node.left, depth + 1, x_start, mid, pos)
    _layout(node.right, depth + 1, mid, x_end, pos)
    pos[id_] = (mid, -depth)
    return mid


def draw_tree(root: ClusterNode, save_path: str = "data/processed_data/cluster_tree.png") -> None:
    """Draw binary cluster tree to PNG."""
    pos = {}
    _layout(root, 0, 0.0, 1.0, pos)
    if not pos:
        _log(f"Tree empty, skipping draw to {save_path}")
        return

    depth = 1 + max(-y for _, y in pos.values())
    fig_h = max(8, min(24, 4 + depth * 1.2))
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_h))
    ax.set_axis_off()

    def draw_edges(node: ClusterNode):
        if node.is_leaf:
            return
        x0, y0 = pos[id(node)]
        for child in (node.left, node.right):
            x1, y1 = pos[id(child)]
            ax.plot([x0, x1], [y0, y1], "k-", lw=1, zorder=0)
            draw_edges(child)

    draw_edges(root)

    id_to_node = {}
    def collect(n):
        id_to_node[id(n)] = n
        if not n.is_leaf:
            collect(n.left)
            collect(n.right)
    collect(root)

    for node_id, (x, y) in pos.items():
        node = id_to_node.get(node_id)
        if node is None:
            continue
        size = node.size()
        ax.scatter([x], [y], s=min(400, 80 + size * 3), c="steelblue", edgecolors="black", zorder=1)
        ax.annotate(str(size), (x, y), ha="center", va="center", fontsize=8, fontweight="bold", zorder=2)

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Tree saved to {save_path}")


def draw_tree_highlight_leaf(
    root: ClusterNode,
    leaf_nodes: list,
    highlight_leaf_id: int,
    save_path: str = "data/processed_data/cluster_tree_current.png",
    title: str = None,
) -> None:
    """Draw cluster tree with one leaf highlighted."""
    pos = {}
    _layout(root, 0, 0.0, 1.0, pos)
    if not pos:
        _log(f"Tree empty, skipping draw to {save_path}")
        return
    highlight_node = leaf_nodes[highlight_leaf_id] if 0 <= highlight_leaf_id < len(leaf_nodes) else None
    highlight_id = id(highlight_node) if highlight_node is not None else None

    depth = 1 + max(-y for _, y in pos.values())
    fig_h = max(8, min(24, 4 + depth * 1.2))
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_h))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)

    def draw_edges(node: ClusterNode):
        if node.is_leaf:
            return
        x0, y0 = pos[id(node)]
        for child in (node.left, node.right):
            x1, y1 = pos[id(child)]
            ax.plot([x0, x1], [y0, y1], "k-", lw=1, zorder=0)
            draw_edges(child)

    draw_edges(root)

    id_to_node = {}
    def collect(n):
        id_to_node[id(n)] = n
        if not n.is_leaf:
            collect(n.left)
            collect(n.right)
    collect(root)

    for node_id, (x, y) in pos.items():
        node = id_to_node.get(node_id)
        if node is None:
            continue
        size = node.size()
        if node_id == highlight_id:
            ax.scatter([x], [y], s=min(500, 100 + size * 3), c="orangered", edgecolors="darkred", linewidths=2, zorder=3)
            ax.annotate(f"{size}\n(current)", (x, y), ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
        else:
            ax.scatter([x], [y], s=min(400, 80 + size * 3), c="steelblue", edgecolors="black", zorder=1)
            ax.annotate(str(size), (x, y), ha="center", va="center", fontsize=8, fontweight="bold", zorder=2)

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)
    ax.set_ylim(min(ys) - 0.2, max(ys) + 0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Tree with current node saved to {save_path}")


def load_titles(csv_path: str = "reddit_posts.csv") -> list:
    """Load title per row; index matches row order."""
    titles = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            titles.append((row.get("title") or "").strip())
    return titles


def print_cluster_samples(leaf_nodes: list, titles: list, samples_per_cluster: int = 6) -> None:
    """Print sample titles per leaf cluster."""
    print("\n--- Cluster samples ---")
    for i, node in enumerate(leaf_nodes):
        inds = node.indices
        size = node.size()
        step = max(1, size // samples_per_cluster) if size > samples_per_cluster else 1
        idx = inds[::step][:samples_per_cluster]
        sample_titles = [titles[j][:80] + ("..." if len(titles[j]) > 80 else "") for j in idx if j < len(titles)]
        print(f"\n  Cluster {i} (size={size})")
        for t in sample_titles:
            print(f"    - {t}")


def get_cluster_labels(vectors: np.ndarray) -> Tuple[np.ndarray, list]:
    """Build cluster tree and return (labels array: doc_idx -> cluster_id, list of leaf nodes)."""
    root = build_cluster_tree(vectors)
    leaf_nodes = leaves(root)
    labels = np.zeros(len(vectors), dtype=np.int64)
    for cid, node in enumerate(leaf_nodes):
        for idx in node.indices:
            labels[idx] = cid
    return labels, leaf_nodes


def run_cluster_tree(
    vectors_path="data/processed_data/doc_vectors.npy",
    csv_path="reddit_posts.csv",
    tree_path="data/processed_data/cluster_tree.png",
    labels_path="data/processed_data/cluster_labels.npy",
):
    """Build cluster tree, draw, save labels and tree images."""
    _log("[CLUSTER] Step 3/3: Loading vectors and building cluster tree...")
    vectors = np.load(vectors_path)
    _log(f"[CLUSTER] Loaded vectors: shape {vectors.shape}")

    _log("[CLUSTER] Building tree (this can take a moment)...")
    root = build_cluster_tree(vectors)
    _log("\n--- Binary cluster tree ---")
    print_tree(root)

    leaf_nodes = leaves(root)
    labels = np.zeros(len(vectors), dtype=np.int64)
    for cid, node in enumerate(leaf_nodes):
        for idx in node.indices:
            labels[idx] = cid
    os.makedirs(os.path.dirname(tree_path) or ".", exist_ok=True)
    np.save(labels_path, labels)
    _log(f"[CLUSTER] Cluster labels saved to {labels_path}")

    print(f"\n--- Summary: {len(leaf_nodes)} leaf clusters ---")
    for i, node in enumerate(leaf_nodes):
        print(f"  Cluster {i}: size {node.size()}")

    out_dir = os.path.dirname(tree_path) or "."
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(tree_path))[0]
    tree_timestamped = os.path.join(out_dir, f"{base}_{stamp}.png")
    draw_tree(root, save_path=tree_timestamped)
    _log(f"[CLUSTER] Full cluster tree saved (old data preserved): {tree_timestamped}")
    try:
        import shutil
        shutil.copy2(tree_timestamped, tree_path)
        _log(f"[CLUSTER] Latest tree copy: {tree_path}")
    except Exception as e:
        _log(f"[CLUSTER] Could not copy to latest: {e}")

    titles = load_titles(csv_path)
    print_cluster_samples(leaf_nodes, titles, samples_per_cluster=6)


def embed_query(text: str, processed_dir: str = "data/processed_data") -> np.ndarray:
    """Embed query with Word2Vec (mean of word vectors); returns vector of same dimension as doc vectors."""
    text = (text or "").strip().lower()
    model_path = os.path.join(processed_dir, "word2vec.model")
    vectors_path = os.path.join(processed_dir, "doc_vectors.npy")

    if os.path.isfile(model_path):
        model = Word2Vec.load(model_path)
        tokens = word_tokenize(text)
        return doc_to_vector(model, tokens).astype(np.float64)

    if os.path.isfile(vectors_path):
        v = np.load(vectors_path)
        return np.zeros(v.shape[1] if v.ndim > 1 else 100, dtype=np.float64)
    return np.zeros(100, dtype=np.float64)


def find_closest_cluster_and_display(
    query: str,
    csv_path: str = "reddit_posts.csv",
    processed_dir: str = "data/processed_data",
    tree_path: str = "data/processed_data/cluster_tree.png",
    tree_current_path: str = "data/processed_data/cluster_tree_current.png",
    max_messages: int = 20,
) -> None:
    """Find closest cluster to query, print sample messages, save tree with node highlighted."""
    vectors = np.load(os.path.join(processed_dir, "doc_vectors.npy"))
    labels = np.load(os.path.join(processed_dir, "cluster_labels.npy"))
    n_clusters = int(labels.max()) + 1
    centroids = np.zeros((n_clusters, vectors.shape[1]))
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = vectors[mask].mean(axis=0)

    q = embed_query(query, processed_dir)
    q = q.reshape(1, -1)
    if q.shape[1] != centroids.shape[1]:
        q = np.hstack([q, np.zeros((1, centroids.shape[1] - q.shape[1]))])[:, :centroids.shape[1]]
    dists = np.linalg.norm(centroids - q, axis=1)
    closest_id = int(np.argmin(dists))

    root = build_cluster_tree(vectors)
    leaf_nodes = leaves(root)
    os.makedirs(os.path.dirname(tree_current_path) or ".", exist_ok=True)
    draw_tree_highlight_leaf(
        root, leaf_nodes, closest_id,
        save_path=tree_current_path,
        title=f"Cluster tree — current node: cluster {closest_id} (query: \"{query[:40]}{'...' if len(query) > 40 else ''}\")",
    )

    titles = load_titles(csv_path)
    indices = np.where(labels == closest_id)[0]
    print(f"\nQuery: \"{query}\"")
    print(f"Closest cluster: {closest_id} ({len(indices)} messages)")
    print("\nMessages from selected cluster:")
    for i, idx in enumerate(indices[:max_messages]):
        if idx < len(titles):
            t = titles[idx]
            print(f"  {i+1}. {t[:90]}{'...' if len(t) > 90 else ''}")
    print(f"\nFull cluster tree: {tree_path}")
    print(f"Tree with current node highlighted: {tree_current_path}")


if __name__ == "__main__":
    _log("data_processing_and_analysis started")
    run_embedding()
    run_cluster_tree()
    _log("data_processing_and_analysis finished")
