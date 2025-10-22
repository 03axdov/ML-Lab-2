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

    If the first non-empty line looks like SGF content (starts with '(' or ';'),
    assume there is no explicit label line in the file and set player_name=None.
    Otherwise, treat the first non-empty line as the label and parse the rest as
    concatenated SGF game records.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        return [], None

    first = lines[0]
    looks_like_sgf = first.startswith("(") or first.startswith(";")
    if looks_like_sgf:
        player_name = None
        text = "\n".join(lines)
    else:
        player_name = first
        text = "\n".join(lines[1:])
    games = _split_top_level_sgf(text)
    return games, player_name


def load_all_games(data_dir):
    """Load all SGF games and label each by its filename (without .sgf)."""
    all_games = []
    all_labels = []

    fnames = [f for f in os.listdir(data_dir) if f.endswith('.sgf')]
    fnames.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    for idx, fname in enumerate(fnames, start=1):
        path = os.path.join(data_dir, fname)
        label = os.path.splitext(fname)[0]  # e.g., '1' for 1.sgf
        print(f"Parsing file {idx}: {fname}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        games = _split_top_level_sgf(text)
        for g in games:
            all_games.append(g)
            all_labels.append(label)

    return all_games, all_labels


# -------------------- Per-move parsing for CNN sequence models --------------------

_MOVE_RE = re.compile(r";([BW])\[([^\]]*)\]")


def extract_moves_from_sgf(sgf_text: str, board_size: int = 19):
    """Extract a list of moves from an SGF game string.

    Returns a list of tuples (player, x, y), where player is 0 for Black and 1 for White.
    Pass moves or malformed coordinates are skipped.
    """
    moves = []
    for m in _MOVE_RE.finditer(sgf_text):
        player_ch, coord = m.group(1), m.group(2)
        if not coord or len(coord) != 2:
            # pass move or malformed; skip
            continue
        x = ord(coord[0]) - ord('a')
        y = ord(coord[1]) - ord('a')
        if 0 <= x < board_size and 0 <= y < board_size:
            player = 0 if player_ch == 'B' else 1
            moves.append((player, x, y))
    return moves


def moves_to_frames(moves, board_size=19, history_k=8, max_moves=120):
    """
    Encode AlphaGo-style board states with safe normalization.

    Each frame represents the current board after a move, plus the last `history_k` board states.
    Planes:
        0..(K-1):  black stones at t-K+1..t
        K..(2K-1): white stones at t-K+1..t
        2K:        player-to-move plane (1=black, 0=white)

    Output shape: (max_moves, board_size, board_size, 2*K + 1)
    """
    import numpy as np

    board = np.zeros((board_size, board_size), dtype=np.int8)
    hist_black, hist_white, frames = [], [], []
    T = min(len(moves), max_moves)

    for i in range(T):
        player, x, y = moves[i]
        board[y, x] = 1 if player == 0 else -1  # 1=black, -1=white

        # record current board as binary planes
        black_plane = (board == 1).astype(np.float32)
        white_plane = (board == -1).astype(np.float32)
        hist_black.append(black_plane)
        hist_white.append(white_plane)

        # pad history
        blk_hist = hist_black[-history_k:]
        wht_hist = hist_white[-history_k:]
        while len(blk_hist) < history_k:
            blk_hist.insert(0, np.zeros((board_size, board_size), np.float32))
        while len(wht_hist) < history_k:
            wht_hist.insert(0, np.zeros((board_size, board_size), np.float32))

        planes = blk_hist + wht_hist

        # add player-to-move plane (1.0 if black to play next)
        next_player = 1.0 if player == 1 else 0.0  # white just moved → black next
        planes.append(np.full((board_size, board_size), next_player, np.float32))

        frames.append(np.stack(planes, axis=-1))

    # pad to max_moves
    if len(frames) < max_moves:
        pad = np.zeros((max_moves - len(frames), board_size, board_size, 2 * history_k + 1), np.float32)
        seq = np.concatenate([np.stack(frames), pad], axis=0)
    else:
        seq = np.stack(frames[:max_moves], axis=0)

    # no normalization; keep true 0/1 sparsity
    return seq.astype(np.float32)
