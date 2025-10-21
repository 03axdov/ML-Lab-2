import os
import re
import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from data_parser import parse_sgf_file, extract_moves_from_sgf, moves_to_frames


def texts_to_padded(texts: List[str], tokenizer, max_len: int) -> np.ndarray:
    seqs = tokenizer.texts_to_sequences(texts)
    return tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=max_len, padding="post", truncating="post"
    )


def compute_embeddings_array(x: np.ndarray, model: tf.keras.Model, batch_size: int = 64) -> np.ndarray:
    embs = model.predict(x, batch_size=batch_size, verbose=0)
    # L2 normalize
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
        p = p / (np.linalg.norm(p) + 1e-12)
        protos[int(c)] = p
    return protos


def predict_by_prototypes(query_embs: np.ndarray, prototypes: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not prototypes:
        return np.array([]), np.array([])
    classes = np.array(sorted(prototypes.keys()))
    P = np.stack([prototypes[int(c)] for c in classes], axis=0)  # [C, D]
    sims = np.matmul(query_embs, P.T)
    pred_ix = sims.argmax(axis=1)
    preds = classes[pred_ix]
    best_sims = sims.max(axis=1)
    return preds, best_sims


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


def run_few_shot(
    cand_dir: str = "data/test_set/cand_set",
    query_dir: str = "data/test_set/query_set",
    embed_model_path: str = "models/embed_model.keras",
    tokenizer_path: str = "cache/tokenizer.pkl",
    max_len: int = 2000,
    k_shots: int = 5,
    batch_size: int = 64,
    out_csv: str = "predictions.csv",
) -> List[Tuple[int, int]]:
    """Run few-shot inference without CLI.

    Returns a list of (query_id, predicted_candidate_id). Also writes a CSV if
    out_csv is provided.
    """
    # Load embedding model (and tokenizer if token-based)
    embed_model = tf.keras.models.load_model(embed_model_path)
    input_shape = embed_model.input_shape
    if isinstance(input_shape, list):
        ishape = input_shape[0]
    else:
        ishape = input_shape
    use_frames = len(ishape) == 5  # [B, T, H, W, C]
    tokenizer = None
    if not use_frames:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

    # Load candidates per file; label = numeric id from filename (e.g., 1..600)
    cand_files = _list_sgf_sorted(cand_dir)
    if not cand_files:
        raise RuntimeError(f"No candidate .sgf files found in {cand_dir}")

    cand_ids: List[int] = []
    cand_all_games: List[str] = []
    cand_all_labels: List[int] = []
    for fname in cand_files:
        cid = _numeric_id_from_name(fname)
        path = os.path.join(cand_dir, fname)
        games, _ = parse_sgf_file(path)
        if not games:
            continue
        # Optionally subsample to K shots per candidate when building prototypes
        if k_shots > 0 and len(games) > k_shots:
            games = games[: k_shots]
        cand_ids.append(cid)
        cand_all_games.extend(games)
        cand_all_labels.extend([cid] * len(games))

    if not cand_all_games:
        raise RuntimeError("No candidate games parsed.")

    # Encode candidates (frames or tokens); build prototype per candidate id
    if use_frames:
        # infer T,H,W,C from model input
        _, T, H, W, C = ishape
        T = int(T) if T is not None else 200
        H = int(H) if H is not None else 19
        history_k = (int(C) - 2) // 2 if C is not None else 4
        cand_frames = []
        for g in cand_all_games:
            mv = extract_moves_from_sgf(g, board_size=H)
            fr = moves_to_frames(mv, board_size=H, history_k=history_k, max_moves=T)
            cand_frames.append(fr)
        X_cand = np.stack(cand_frames, axis=0)
    else:
        X_cand = texts_to_padded(cand_all_games, tokenizer, max_len)
    E_cand = compute_embeddings_array(X_cand, embed_model, batch_size=batch_size)
    cand_all_labels = np.array(cand_all_labels, dtype=np.int32)
    prototypes = build_prototypes(E_cand, cand_all_labels, k=0)  # already limited by K shots
    if not prototypes:
        raise RuntimeError("Failed to build any prototypes from candidates.")

    # Load queries per file; output one prediction per query file id
    query_files = _list_sgf_sorted(query_dir)
    if not query_files:
        raise RuntimeError(f"No query .sgf files found in {query_dir}")

    results: List[Tuple[int, int]] = []  # (query_id, pred_candidate_id)
    for fname in query_files:
        qid = _numeric_id_from_name(fname)
        path = os.path.join(query_dir, fname)
        games, _ = parse_sgf_file(path)
        if not games:
            continue
        if use_frames:
            query_frames = []
            # reuse inferred T,H from above
            for g in games:
                mv = extract_moves_from_sgf(g, board_size=H)
                fr = moves_to_frames(mv, board_size=H, history_k=history_k, max_moves=T)
                query_frames.append(fr)
            Xq = np.stack(query_frames, axis=0)
        else:
            Xq = texts_to_padded(games, tokenizer, max_len)
        Eq = compute_embeddings_array(Xq, embed_model, batch_size=batch_size)
        # Aggregate a query prototype by averaging all games in the file
        q_proto = Eq.mean(axis=0)
        q_proto = q_proto / (np.linalg.norm(q_proto) + 1e-12)
        pred_id_arr, _ = predict_by_prototypes(q_proto[None, :], prototypes)
        pred_cid = int(pred_id_arr[0])
        results.append((qid, pred_cid))

    # Write CSV with columns id,label sorted by id
    results.sort(key=lambda x: x[0])
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("id,label\n")
            for qid, pred in results:
                f.write(f"{qid},{pred}\n")

    print(f"Candidates: {len(prototypes)} | Queries: {len(results)}")
    if out_csv:
        print(f"Wrote predictions to {out_csv}")
    return results


if __name__ == "__main__":
    run_few_shot()
