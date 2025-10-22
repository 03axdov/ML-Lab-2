import os
import glob
import numpy as np

OUT_DIR = "data/processed"

def main():
    label_path = os.path.join(OUT_DIR, "label_classes.npy")
    shard_paths = sorted(glob.glob(os.path.join(OUT_DIR, "games_shard_*.npz")))[:5]

    if not shard_paths:
        print("❌ No shards found. Run preprocessing first.")
        return
    if not os.path.exists(label_path):
        print("❌ label_classes.npy not found.")
        return

    # --- 1️⃣ Load label mapping ---
    classes = np.load(label_path, allow_pickle=True)
    num_classes = len(classes)
    print(f"✅ Found label_classes.npy with {num_classes} classes.")
    print("   First few labels:", classes[:10])

    all_labels = []
    shapes = set()
    total_samples = 0

    # --- 2️⃣ Inspect shards ---
    for f in shard_paths:
        try:
            data = np.load(f)
            X, y = data["X"], data["y"]
            all_labels.extend(y.tolist())
            shapes.add(X.shape[1:])
            total_samples += len(y)

            print(f"{os.path.basename(f):25s} "
                  f"y.shape={y.shape}  X.shape={X.shape} "
                  f"min={X.min():.3f} max={X.max():.3f} mean={X.mean():.4f} std={X.std():.4f} "
                  f"unique_labels={len(np.unique(y))}")

        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")

    # --- 3️⃣ Summary ---
    all_labels = np.array(all_labels)
    unique_labels = np.unique(all_labels)

    print("\n--- Summary ---")
    print(f"Total shards: {len(shard_paths)}")
    print(f"Total samples: {total_samples}")
    print(f"Unique label IDs: {len(unique_labels)}/{num_classes}")
    print(f"Label ID range: {all_labels.min()}–{all_labels.max()}")
    print(f"Unique frame shapes: {shapes}")

    # --- 4️⃣ Sanity checks ---
    if all_labels.min() == 0 and all_labels.max() == num_classes - 1:
        print("✅ Label IDs cover full 0–N−1 range.")
    else:
        print("❌ Label IDs not covering full range! Check label encoding.")

    # --- 5️⃣ Random sample inspection ---
    import random
    rid = random.randint(0, len(shard_paths) - 1)
    data = np.load(shard_paths[rid])
    X, y = data["X"], data["y"]
    i = random.randint(0, len(y) - 1)
    frame = X[i]
    print(f"\nRandom sample from {os.path.basename(shard_paths[rid])}:")
    print(f"  Label ID: {y[i]} → name: {classes[y[i]]}")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame min/max: {frame.min():.3f}/{frame.max():.3f}")
    print(f"  Fraction non-zero: {np.count_nonzero(frame) / frame.size:.4f}")

    print("\n✅ Preprocessing verification complete.")


if __name__ == "__main__":
    main()
