
def state_3x3_to_flat(state):
    """
    Convert a numeric 3x3 state (tuple of tuples) to a flat 9-element list
    of strings ("X", "O", " ") so it can be passed to opponents.py agents.
    Mapping: 1 -> "X", -1 -> "O", 0 -> " "
    """
    mapping = {1: "X", -1: "O", 0: " "}
    return [mapping[cell] for row in state for cell in row]


def flat_to_state_3x3(flat):
    """
    Convert a flat 9-element list/tuple of strings back to a numeric 3x3
    tuple of tuples so it can be used with tictactoe.py / training.py.
    Mapping: "X" -> 1, "O" -> -1, " " -> 0
    """
    mapping = {"X": 1, "O": -1, " ": 0}
    values = [mapping[v] for v in flat]
    return tuple(tuple(values[i*3:(i+1)*3]) for i in range(3))


def flat_action_to_rowcol(action_idx):
    """Convert a flat integer index (0-8) to (row, col)."""
    return (action_idx // 3, action_idx % 3)


def rowcol_to_flat_action(row, col):
    """Convert (row, col) to a flat integer index (0-8)."""
    return row * 3 + col



class OpponentAdapter:
    """
    Wraps a opponents.py-style agent (choose_action interface) so it can be
    used anywhere training.py expects an act(state, actions) interface.
    """

    def __init__(self, opponent):
        self.opponent = opponent

    def act(self, state, actions):
        flat = state_3x3_to_flat(state)
        flat_action = self.opponent.choose_action(flat)
        return flat_action_to_rowcol(flat_action)



_WIN_LINES = [
    # rows
    ((0,0),(0,1),(0,2)),
    ((1,0),(1,1),(1,2)),
    ((2,0),(2,1),(2,2)),
    # cols
    ((0,0),(1,0),(2,0)),
    ((0,1),(1,1),(2,1)),
    ((0,2),(1,2),(2,2)),
    # diagonals
    ((0,0),(1,1),(2,2)),
    ((0,2),(1,1),(2,0)),
]


def _line_score(state, line, player):
    """
    Count how many cells in `line` are occupied by `player` and how many
    are empty.  Returns (player_count, empty_count).
    Returns (-1, -1) if the line is blocked by the opponent.
    """
    pc = ec = 0
    for (r, c) in line:
        v = state[r][c]
        if v == player:
            pc += 1
        elif v == 0:
            ec += 1
        else:
            return -1, -1
    return pc, ec


def _count_threats(state, player):
    """
    Count lines where `player` has 2 pieces and 1 empty (immediate win threat).
    """
    count = 0
    for line in _WIN_LINES:
        pc, ec = _line_score(state, line, player)
        if pc == 2 and ec == 1:
            count += 1
    return count


def _count_open_lines(state, player):
    """Count lines not blocked by the opponent (player can still win on them)."""
    count = 0
    for line in _WIN_LINES:
        pc, ec = _line_score(state, line, player)
        if pc >= 0:
            count += 1
    return count


def _is_winning_state(state, player):
    """Return True if `player` has already won in `state`."""
    for line in _WIN_LINES:
        if all(state[r][c] == player for (r, c) in line):
            return True
    return False


def _is_draw(state):
    """Return True if every cell is filled and nobody has won."""
    if any(state[r][c] == 0 for r in range(3) for c in range(3)):
        return False
    return not (_is_winning_state(state, 1) or _is_winning_state(state, -1))


def extract_features(state, action):
    """
    Return a list of (feature_key, value) pairs for the given state-action.

    Features cover:
      - Bias
      - Action location (centre / corner / edge)
      - Immediate win / block flags
      - Threat counts for both players before and after
      - Open-line counts
      - Board occupancy
      - Max line length through the action cell
      - Fork detection
    """
    row, col = action
    flat = [state[r][c] for r in range(3) for c in range(3)]
    player   = 1
    opponent = -1

    features = []

    # 1. Bias
    features.append(("bias", 1))

    # 2. Action position type
    is_centre = (row == 1 and col == 1)
    is_corner = (row in (0, 2) and col in (0, 2))
    is_edge   = not is_centre and not is_corner

    features.append(("pos_centre", int(is_centre)))
    features.append(("pos_corner", int(is_corner)))
    features.append(("pos_edge",   int(is_edge)))

    # 3. Exact action cell (lets agent learn cell-specific values)
    features.append(("action_rc", (row, col)))

    # 4. Simulate placing our piece at action
    sim = [list(r) for r in state]
    sim[row][col] = player
    sim_state = tuple(tuple(r) for r in sim)

    # 5. Immediate win?
    immediate_win = _is_winning_state(sim_state, player)
    features.append(("immediate_win", int(immediate_win)))

    # 6. Does this block the opponent from winning immediately?
    sim2 = [list(r) for r in state]
    sim2[row][col] = opponent
    sim2_state = tuple(tuple(r) for r in sim2)
    blocks_opponent = _is_winning_state(sim2_state, opponent)
    features.append(("blocks_opponent", int(blocks_opponent)))

    # 7. Threat counts before action (capped to reduce state space)
    my_threats  = min(_count_threats(state, player),   3)
    opp_threats = min(_count_threats(state, opponent), 3)
    features.append(("my_threats_before",  my_threats))
    features.append(("opp_threats_before", opp_threats))

    # 8. Threat counts after action
    my_threats_after = min(_count_threats(sim_state, player), 3)
    features.append(("my_threats_after", my_threats_after))

    # 9. Open lines for both players
    features.append(("my_open_lines",  _count_open_lines(state, player)))
    features.append(("opp_open_lines", _count_open_lines(state, opponent)))

    # 10. Board occupancy (game phase)
    filled = sum(1 for v in flat if v != 0)
    features.append(("filled_cells", filled))

    # 11. Max piece-count along any line passing through action cell
    lines_through = [ln for ln in _WIN_LINES if (row, col) in ln]
    my_max_line  = 0
    opp_max_line = 0
    for line in lines_through:
        pc, _ = _line_score(state, line, player)
        if pc >= 0:
            my_max_line = max(my_max_line, pc)
        opc, _ = _line_score(state, line, opponent)
        if opc >= 0:
            opp_max_line = max(opp_max_line, opc)

    features.append(("my_max_line_through",  my_max_line))
    features.append(("opp_max_line_through", opp_max_line))

    # 12. Fork: action creates 2+ simultaneous threats
    features.append(("creates_fork", int(my_threats_after >= 2)))

    return features



#  Reward function
def reward_function(state, action, next_state):
    """
    Compute a shaped reward for the transition state -> (action) -> next_state.
    """
    player   = 1
    opponent = -1

    # Terminal: our agent won
    if _is_winning_state(next_state, player):
        return 1.0

    # Terminal: opponent won (next_state shows -1 has a full line)
    if _is_winning_state(next_state, opponent):
        return -1.0

    # Draw
    if _is_draw(next_state):
        return -0.05

    # Non-terminal shaping
    reward = 0.0

    # Reward increasing our threats
    my_threats_after  = _count_threats(next_state, player)
    my_threats_before = _count_threats(state,      player)
    reward += 0.05 * (my_threats_after - my_threats_before)

    # Reward reducing opponent threats
    opp_threats_after  = _count_threats(next_state, opponent)
    opp_threats_before = _count_threats(state,      opponent)
    if opp_threats_before > 0 and opp_threats_after < opp_threats_before:
        reward += 0.03 * (opp_threats_before - opp_threats_after)

    # Small positional bonuses
    row, col = action
    if row == 1 and col == 1 and state[1][1] == 0:
        reward += 0.02   # centre
    elif (row, col) in ((0,0),(0,2),(2,0),(2,2)) and state[row][col] == 0:
        reward += 0.01   # corner

    return reward



#  Epsilon schedule

def epsilon_schedule(episode):
    """
    Return the exploration rate epsilon for the given episode.

    Starts at 0.9, decays at 0.995 per episode, minimum 0.05.
    """
    return max(0.05, 0.9 * (0.995 ** episode))