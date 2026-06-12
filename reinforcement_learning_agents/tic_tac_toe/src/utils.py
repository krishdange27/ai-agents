# utils.py
# Board representation helpers and opponent interface adapter.
#
# The environment uses numeric 3x3 states (tuple of tuples, 1/-1/0).
# The opponents module uses flat 9-element string lists ("X"/"O"/" ").
# These utilities bridge the two.

def state_3x3_to_flat(state):
    """
    Convert a numeric 3x3 board to a flat 9-element string list.
        1  -> "X"  (agent)
       -1  -> "O"  (opponent)
        0  -> " "  (empty)
    """
    mapping = {1: "X", -1: "O", 0: " "}
    return [mapping[cell] for row in state for cell in row]


def flat_to_state_3x3(flat):
    """
    Convert a flat 9-element string list back to a numeric 3x3 tuple.
        "X" -> 1
        "O" -> -1
        " " -> 0
    """
    mapping = {"X": 1, "O": -1, " ": 0}
    values = [mapping[v] for v in flat]
    return tuple(tuple(values[i*3:(i+1)*3]) for i in range(3))


def flat_index_to_rowcol(idx):
    """Convert flat board index (0-8) to (row, col)."""
    return (idx // 3, idx % 3)


def rowcol_to_flat_index(row, col):
    """Convert (row, col) to flat board index (0-8)."""
    return row * 3 + col


def board_to_string(state):
    """
    Render a 3x3 state as a human-readable string.
    Used by demo scripts to print board positions.
    """
    symbols = {1: "X", -1: "O", 0: "."}
    rows = []
    for row in state:
        rows.append(" | ".join(symbols[v] for v in row))
    divider = "\n--+---+--\n"
    return divider.join(rows)


class OpponentAdapter:
    """
    Wraps an opponents.py-style agent for use in training.py.

    opponents.py interface:  choose_action(flat_string_state) -> int index
    training.py interface:   act(state_3x3, valid_actions)    -> (row, col)
    """

    def __init__(self, opponent):
        self.opponent = opponent

    def act(self, state, actions):
        flat = state_3x3_to_flat(state)
        flat_action = self.opponent.choose_action(flat)
        return flat_index_to_rowcol(flat_action)
