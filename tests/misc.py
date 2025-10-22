import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load encoder
classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
print("Num classes in encoder:", len(classes))

# Load one shard
data = np.load("data/processed/games_shard_000.npz")
y = data["y"]
print("Label range in shard:", y.min(), y.max())
print("Example IDs:", y[:10])

# Recreate encoder mapping
le = LabelEncoder()
le.classes_ = classes

# Decode first few integer IDs back to original filenames
decoded = le.inverse_transform(y[:10])
print("Decoded filenames:", decoded)
