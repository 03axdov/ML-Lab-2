# fewshot_diagnostic.py
# Creates a few-shot sanity test from known players and evaluates accuracy
# against your existing few_shot_infer.run_few_shot() function.
#
# NOTE: If your few_shot_infer.py imports from "data_parser" but your code lives
# under "game_parser.py", either create a small alias module named data_parser.py
# that re-exports the same functions, or change that import in few_shot_infer.py.

import os
import random
import argparse
from typing import Tuple, List

# Use the same splitter as in your parser
from data_parser import _split_top_level_sgf
from few_shot_infer import run_few_shot


def make_known_player_fewshot(
    source_dir: str,
    out_root: str,
    n_players: int,
    n_cand_games: int,
    n_query_games: int,
    seed: int = 42,
) -> Tuple[str, str, List[int]]:
    """
    Build a few-shot sanity set from players that were seen in training.

    Each selected player contributes n_cand_games to candidate set and
    n_query_games to query set. Files are written as <player_id>.sgf where
    <player_id> is the original filename stem (e.g., "123" from "123.sgf").
    """
    random.seed(seed)
    cand_dir = os.path.join(out_root, "cand_set")
    query_dir = os.path.join(out_root, "query_set")
    os.makedirs(cand_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    sgf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".sgf")]
    # Sort numerically if filenames are numbers like "1.sgf", "2.sgf", ...
    def _key(f):
        stem = os.path.splitext(f)[0]
        return int(stem) if stem.isdigit() else stem
    sgf_files.sort(key=_key)

    chosen = random.sample(sgf_files, min(n_players, len(sgf_files)))
    kept_player_ids: List[int] = []

    for f in chosen:
        stem = os.path.splitext(f)[0]
        try:
            pid = int(stem)
        except ValueError:
            # Skip non-numeric filenames; adjust if your filenames are not numeric
            continue

        src = os.path.join(source_dir, f)
        with open(src, "r", encoding="utf-8") as fh:
            text = fh.read()
        games = _split_top_level_sgf(text)

        if len(games) < n_cand_games + n_query_games:
            # Not enough games for both sets; skip
            continue

        random.shuffle(games)
        cand_games = games[:n_cand_games]
        query_games = games[n_cand_games:n_cand_games + n_query_games]

        with open(os.path.join(cand_dir, f"{pid}.sgf"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(cand_games))
        with open(os.path.join(query_dir, f"{pid}.sgf"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(query_games))

        kept_player_ids.append(pid)

    print(f"Built few-shot check sets in {out_root}")
    print(f"  Candidates: {len(os.listdir(cand_dir))} | Queries: {len(os.listdir(query_dir))}")
    return cand_dir, query_dir, kept_player_ids


def main():
    p = argparse.ArgumentParser(description="Few-shot diagnostic on known players")
    p.add_argument("--source_dir", type=str, default="data/train_set",
                   help="Folder with training player .sgf files (one file per player).")
    p.add_argument("--out_root", type=str, default="data/fewshot_check",
                   help="Output root folder for cand_set/ and query_set/ plus CSV.")
    p.add_argument("--n_players", type=int, default=20,
                   help="How many players to sample for the sanity check.")
    p.add_argument("--n_cand_games", type=int, default=5,
                   help="Games per player used to form the prototype.")
    p.add_argument("--n_query_games", type=int, default=5,
                   help="Games per player used as queries.")
    p.add_argument("--k_shots", type=int, default=5,
                   help="few_shot_infer's k_shots; usually match n_cand_games.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for embedding inference in few_shot_infer.")
    args = p.parse_args()

    cand_dir, query_dir, kept_ids = make_known_player_fewshot(
        source_dir=args.source_dir,
        out_root=args.out_root,
        n_players=args.n_players,
        n_cand_games=args.n_cand_games,
        n_query_games=args.n_query_games,
        seed=42,
    )

    # Ensure CSV goes under out_root so path is consistent
    out_csv = os.path.join(args.out_root, "fewshot_check_predictions.csv")

    # Call your few-shot pipeline directly (no subprocess)
    results = run_few_shot(
        cand_dir=cand_dir,
        query_dir=query_dir,
        k_shots=args.k_shots,
        batch_size=args.batch_size,
        out_csv=out_csv,
    )
    # results is a List[Tuple[qid, pred_id]]

    # Compute accuracy directly (ground truth for this sanity check is qid == true player id)
    if not results:
        print("No results produced. Check that candidate/query sets contain games.")
        return

    correct = sum(1 for qid, pred in results if qid == pred)
    total = len(results)
    acc = correct / total
    print(f"Few-shot accuracy on KNOWN players: {acc*100:.2f}%  ({correct}/{total})")

    # Small report of first few mismatches
    mistakes = [(qid, pred) for qid, pred in results if qid != pred]
    if mistakes:
        print("First 10 mismatches (qid -> pred):")
        for qid, pred in mistakes[:10]:
            print(f"  {qid} -> {pred}")

    print(f"CSV written to: {out_csv}")


if __name__ == "__main__":
    main()
