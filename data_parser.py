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


def moves_to_frames(moves, board_size: int = 19, history_k: int = 4, max_moves: int = 200):
    """Convert a move list into a sequence of spatial frames for CNN processing.

    Each frame has shape (board_size, board_size, C) with C = 2*history_k + 2 planes:
      - history_k planes for last k Black moves (one-hot)
      - history_k planes for last k White moves (one-hot)
      - 1 plane for current move one-hot
      - 1 plane filled with 1.0 if current mover is Black else 0.0

    The sequence is padded with zeros up to max_moves.
    """
    C = 2 * history_k + 2
    T = min(len(moves), max_moves)
    frames = []
    last_b = []
    last_w = []

    for i in range(T):
        player, x, y = moves[i]
        planes = []
        # Build history planes for Black
        for k in range(history_k):
            plane = [[0.0] * board_size for _ in range(board_size)]
            if k < len(last_b):
                bx, by = last_b[-1 - k]
                plane[by][bx] = 1.0
            planes.append(plane)
        # History planes for White
        for k in range(history_k):
            plane = [[0.0] * board_size for _ in range(board_size)]
            if k < len(last_w):
                wx, wy = last_w[-1 - k]
                plane[wy][wx] = 1.0
            planes.append(plane)
        # Current move one-hot
        cur = [[0.0] * board_size for _ in range(board_size)]
        cur[y][x] = 1.0
        planes.append(cur)
        # Player plane (all ones for Black, zeros for White)
        player_plane_val = 1.0 if player == 0 else 0.0
        player_plane = [[player_plane_val] * board_size for _ in range(board_size)]
        planes.append(player_plane)

        # Update history buffers
        if player == 0:
            last_b.append((x, y))
        else:
            last_w.append((x, y))

        # Convert to numpy HWC
        import numpy as _np

        frame = _np.stack([_np.array(p, dtype=_np.float32) for p in planes], axis=-1)  # HWC
        frames.append(frame)

    import numpy as _np

    if frames:
        seq = _np.stack(frames, axis=0)  # THWC
    else:
        seq = _np.zeros((0, board_size, board_size, C), dtype=_np.float32)

    # Pad to max_moves
    if seq.shape[0] < max_moves:
        pad = _np.zeros((max_moves - seq.shape[0], board_size, board_size, C), dtype=_np.float32)
        seq = _np.concatenate([seq, pad], axis=0)
    elif seq.shape[0] > max_moves:
        seq = seq[:max_moves]

    return seq.astype(_np.float32)
