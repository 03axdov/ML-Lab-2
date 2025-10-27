import os
import re
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
from tensorflow.keras import mixed_precision

from data_parser import parse_sgf_file, extract_moves_from_sgf, moves_to_frames

mixed_precision.set_global_policy("mixed_float16")

# ---------- GPU setup ----------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Enabled memory growth on {len(gpus)} GPU(s).")
    except RuntimeError:
        pass


# ---------- Embedding helpers ----------

@tf.function
def _embed_step(batch, model):
    """Fast compiled embedding call."""
    return model(batch, training=False)


def compute_embeddings_array(x: np.ndarray, model: tf.keras.Model, batch_size: int = 64) -> np.ndarray:
    """Compute L2-normalized embeddings efficiently on GPU."""
    embs = []
    for start in range(0, len(x), batch_size):
        end = start + batch_size
        xb = tf.convert_to_tensor(x[start:end], dtype=tf.float16)  # mixed precision inference
        eb = _embed_step(xb, model)
        embs.append(eb.numpy())
        del xb, eb
    embs = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms


def build_prototypes(embs: np.ndarray, labels: np.ndarray, k: int = 5, seed: int = 42) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    protos: Dict[int, np.ndarray] = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) > k > 0:
            idx = rng.choice(idx, size=k, replace=False)
        p = embs[idx].mean(axis=0)
        protos[int(c)] = p / (np.linalg.norm(p) + 1e-12)
    return protos


def predict_by_prototypes(query_embs: np.ndarray, prototypes: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not prototypes:
        return np.array([]), np.array([])
    classes = np.array(sorted(prototypes.keys()))
    P = np.stack([prototypes[int(c)] for c in classes], axis=0)
    sims = np.matmul(query_embs, P.T)
    pred_ix = sims.argmax(axis=1)
    preds = classes[pred_ix]
    best_sims = sims.max(axis=1)
    return preds, best_sims


# ---------- File utilities ----------

def _numeric_id_from_name(name: str) -> int:
    base = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r"(\d+)", base)
    if not m:
        raise ValueError(f"Cannot parse numeric id from filename '{name}'")
    return int(m.group(1))


def _list_sgf_sorted(dir_path: str) -> List[str]:
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(".sgf")]
    files.sort(key=lambda x: _numeric_id_from_name(x))
    return files


# ---------- Few-shot inference ----------

def run_few_shot(
    cand_dir: str = "data/test_set/cand_set",
    query_dir: str = "data/test_set/query_set",
    embed_model_path: str = "models/embed_model.keras",
    classifier_model_path: str = "models/cls_model.keras",
    k_shots: int = 5,
    batch_size: int = 8,  # small batch for GPU safety
    out_csv: str = "predictions.csv",
) -> List[Tuple[int, int]]:

    # --- Load or create embedding model ---
    if os.path.exists(embed_model_path):
        print(f"🧠 Loading embedding model: {embed_model_path}")
        embed_model = tf.keras.models.load_model(embed_model_path)
    elif os.path.exists(classifier_model_path):
        print(f"⚙️ Extracting embedding submodel from {classifier_model_path} ...")
        base_model = tf.keras.models.load_model(classifier_model_path)
        try:
            out = base_model.get_layer("embedding").output
        except Exception:
            out = base_model.layers[-2].output  # fallback for older checkpoints
        embed_model = tf.keras.Model(inputs=base_model.input, outputs=out)
        os.makedirs(os.path.dirname(embed_model_path), exist_ok=True)
        embed_model.save(embed_model_path)
        print(f"✅ Saved embedding model to {embed_model_path}")
    else:
        raise FileNotFoundError("Neither embed_model.keras nor cls_model.keras found.")

    _, T, H, W, C = embed_model.input_shape
    history_k = (C - 1) // 2

    # --- Candidates ---
    cand_files = _list_sgf_sorted(cand_dir)
    cand_ids, frames, labels = [], [], []
    print(f"📚 Encoding candidate games ({len(cand_files)} files)...")

    for fname in cand_files:
        cid = _numeric_id_from_name(fname)
        games, _ = parse_sgf_file(os.path.join(cand_dir, fname))
        if not games:
            continue
        if k_shots > 0 and len(games) > k_shots:
            games = games[:k_shots]
        for g in games:
            mv = extract_moves_from_sgf(g, board_size=H)
            fr = moves_to_frames(mv, board_size=H, history_k=history_k, max_moves=T)
            frames.append(fr)
            labels.append(cid)
        cand_ids.append(cid)

    labels = np.array(labels, dtype=np.int32)
    print(f"🧩 Computing candidate embeddings on GPU...")
    E_cand = compute_embeddings_array(np.array(frames, dtype=np.float16), embed_model, batch_size)
    prototypes = build_prototypes(E_cand, labels, k=0)
    del frames, E_cand

    # --- Queries ---
    query_files = _list_sgf_sorted(query_dir)
    results: List[Tuple[int, int]] = []
    print(f"🔍 Predicting {len(query_files)} queries...")

    for fname in query_files:
        qid = _numeric_id_from_name(fname)
        print(f"Processing {qid}")
        games, _ = parse_sgf_file(os.path.join(query_dir, fname))
        if not games:
            continue
        q_frames = []
        for g in games:
            mv = extract_moves_from_sgf(g, board_size=H)
            fr = moves_to_frames(mv, board_size=H, history_k=history_k, max_moves=T)
            q_frames.append(fr)
        Xq = np.array(q_frames, dtype=np.float16)
        Eq = compute_embeddings_array(Xq, embed_model, batch_size)
        q_proto = Eq.mean(axis=0)
        q_proto /= (np.linalg.norm(q_proto) + 1e-12)
        pred_id, _ = predict_by_prototypes(q_proto[None, :], prototypes)
        results.append((qid, int(pred_id[0])))

    # --- Save ---
    results.sort(key=lambda x: x[0])
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("id,label\n")
            for qid, pred in results:
                f.write(f"{qid},{pred}\n")

    print(f"✅ Done. Candidates: {len(prototypes)} | Queries: {len(results)}")
    if out_csv:
        print(f"📁 Predictions saved to {out_csv}")
    return results


if __name__ == "__main__":
    run_few_shot()
