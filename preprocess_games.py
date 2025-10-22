import os
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from sklearn.preprocessing import LabelEncoder
from data_parser import load_all_games, extract_moves_from_sgf, moves_to_frames

# ---------------- CONFIG ----------------
BOARD_SIZE = 19
HISTORY_K = 3
MAX_MOVES = 120
SHARD_SIZE = 2000           # keep smaller to reduce memory (each ~2.3 GB float32)
CHUNK_SIZE = 300            # number of parallel jobs in flight
MAX_WORKERS = os.cpu_count() or 8
DATA_DIR = "data/train_set"
OUT_DIR = "data/processed"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def process_one_game(args):
    """Worker: parse SGF → frame tensor."""
    sgf, label = args
    try:
        moves = extract_moves_from_sgf(sgf, board_size=BOARD_SIZE)
        frames = moves_to_frames(
            moves,
            board_size=BOARD_SIZE,
            history_k=HISTORY_K,
            max_moves=MAX_MOVES
        ).astype(np.float32)
        return frames, label
    except Exception as e:
        print(f"⚠️ Error processing game: {e}")
        return None


def chunked_iterable(it, size):
    """Yield successive chunks from iterable."""
    it = iter(it)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def save_shard_streaming(X_buf, y_buf, shard):
    """Stream each sample to a temporary file, then combine using memmap to avoid large RAM use."""
    shard_path = os.path.join(OUT_DIR, f"games_shard_{shard:03d}.npz")

    # Write one sample at a time into a memmap file
    n = len(X_buf)
    shape = X_buf[0].shape
    mmap_path = shard_path + ".tmp.npy"

    X_mm = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=(n, *shape))
    for i, x in enumerate(X_buf):
        X_mm[i] = x
    del X_mm  # flush to disk

    # Now reopen memory-mapped for compression
    X = np.memmap(mmap_path, dtype=np.float32, mode='r', shape=(n, *shape))
    y = np.array(y_buf, dtype=np.int64)
    np.savez_compressed(shard_path, X=X, y=y)
    del X
    os.remove(mmap_path)
    print(f"✅ Saved shard {shard:03d} ({n} samples)")
    return shard + 1


def main():
    print("Loading SGF files and labels...")
    games, labels = load_all_games(DATA_DIR)
    print(f"Loaded {len(games)} games from {len(set(labels))} unique labels")

    # ---- 1. Fit global encoder ----
    le = LabelEncoder()
    le.fit(labels)
    np.save(os.path.join(OUT_DIR, "label_classes.npy"), le.classes_)
    y_all = le.transform(labels)
    num_classes = len(le.classes_)
    print(f"Encoded {num_classes} unique labels.")

    # ---- 2. Shuffle before chunking ----
    combined = list(zip(games, y_all))
    random.shuffle(combined)
    games, y_all = zip(*combined)
    del combined

    # ---- 3. Process with multiprocessing ----
    print(f"Processing with {MAX_WORKERS} workers...")
    shard, processed = 0, 0
    X_buf, y_buf = [], []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for chunk in chunked_iterable(zip(games, y_all), CHUNK_SIZE):
            futures = [executor.submit(process_one_game, job) for job in chunk]

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                frames, label = result
                X_buf.append(frames)
                y_buf.append(label)
                processed += 1

                if len(X_buf) >= SHARD_SIZE:
                    shard = save_shard_streaming(X_buf, y_buf, shard)
                    X_buf, y_buf = [], []

            if processed % 1000 == 0:
                print(f"Processed {processed}/{len(games)} games...")

    if X_buf:
        shard = save_shard_streaming(X_buf, y_buf, shard)

    print("🎉 Finished preprocessing all games.")
    print(f"Shards and label_classes.npy saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
