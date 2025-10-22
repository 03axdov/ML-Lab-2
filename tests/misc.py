import numpy as np

data = np.load("data/processed/games_shard_000.npz")
X, y = data["X"], data["y"]
print("Global min/max:", X.min(), X.max())
print("Mean:", X.mean(), "Std:", X.std())

# Check how many moves actually contain stones
nonzero = np.count_nonzero(np.abs(X) > 0)
total = np.prod(X.shape)
print("Fraction non-zero:", nonzero / total)