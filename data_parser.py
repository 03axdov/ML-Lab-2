import os
import re
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def _split_top_level_sgf(text: str):
    """Split concatenated SGF records into top-level games.

    SGF uses parentheses to denote variation trees; top-level game records are
    also wrapped in parentheses. This splitter scans the text and collects
    substrings where the parenthesis depth returns to zero, while ignoring
    parentheses encountered inside property values (which are inside square
    brackets). It also respects escaped closing brackets within values.
    """
    games = []
    depth = 0
    in_value = False  # inside [...] property value
    escape = False
    start_idx = None

    for i, ch in enumerate(text):
        if in_value:
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == ']':
                in_value = False
            continue

        # not in value
        if ch == '[':
            in_value = True
            escape = False
            continue

        if ch == '(':
            if depth == 0:
                start_idx = i
            depth += 1
            continue

        if ch == ')':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    games.append(text[start_idx:i + 1])
                    start_idx = None
            continue

    return games


def parse_sgf_file(filepath):
    """Parse a single .sgf file and return (games, player_name).

    The first non-empty line is expected to be the player label; the rest of
    the file contains concatenated SGF game records.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        return [], None

    player_name = lines[0]
    text = "\n".join(lines[1:])
    games = _split_top_level_sgf(text)
    return games, player_name


def load_all_games(data_dir):
    """Load all games and their player labels.

    Returns two lists of equal length: one SGF string per game and the
    corresponding label (player name) for each game.
    """
    all_games = []
    all_labels = []

    # Sort files numerically if possible (e.g., "1.sgf".."200.sgf") for stability
    fnames = [f for f in os.listdir(data_dir) if f.endswith('.sgf')]
    fnames.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    for idx, fname in enumerate(fnames, start=1):
        print(f"Parsing file {idx}: {fname}")
        path = os.path.join(data_dir, fname)
        games, player_name = parse_sgf_file(path)
        for g in games:
            all_games.append(g)
            all_labels.append(player_name)

    return all_games, all_labels
