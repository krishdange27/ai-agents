# agent.py
# Q-learning agent design for TicTacToe.
#
# Three components define the agent's behaviour:
#   - extract_features : what the agent perceives about a state-action pair
#   - reward_function  : what the agent is trying to maximise
#   - epsilon_schedule : how exploration decays over training
#
# The Q-function is linear:  Q(s, a) = w^T * phi(s, a)
# Weights are updated via TD learning in agents.py.

# ── Board analysis helpers ────────────────────────────────────────────────────

_WIN_LINES = [
    ((0,0),(0,1),(0,2)), ((1,0),(1,1),(1,2)), ((2,0),(2,1),(2,2)),  # rows
    ((0,0),(1,0),(2,0)), ((0,1),(1,1),(2,1)), ((0,2),(1,2),(2,2)),  # cols
    ((0,0),(1,1),(2,2)), ((0,2),(1,1),(2,0)),                        # diagonals
]


def _line_score(state, line, player):
    """
    Returns (player_count, empty_count) for a line.
    Returns (-1, -1) if the line is blocked by the opponent.
    """
    pc = ec = 0
    for (r, c) in line:
        v = state[r][c]
        if v == player:       pc += 1
        elif v == 0:          ec += 1
        else:                 return -1, -1
    return pc, ec


def _count_threats(state, player):
    """Lines where player has 2 pieces and 1 empty — immediate win available."""
    return sum(1 for ln in _WIN_LINES if _line_score(state, ln, player) == (2, 1))


def _count_open_lines(state, player):
    """Lines not yet blocked by the opponent."""
    return sum(1 for ln in _WIN_LINES if _line_score(state, ln, player)[0] >= 0)


def _is_winner(state, player):
    """True if player occupies a full winning line."""
    return any(all(state[r][c] == player for r, c in ln) for ln in _WIN_LINES)


def _is_draw(state):
    """True if the board is full with no winner."""
    if any(state[r][c] == 0 for r in range(3) for c in range(3)):
        return False
    return not (_is_winner(state, 1) or _is_winner(state, -1))


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(state, action):
    """
    Build the feature vector phi(s, a) for the linear Q-function.

    14 features capturing:
      - Positional type of the action (centre / corner / edge)
      - Exact cell identity
      - Immediate win and block detection
      - Threat counts before and after placing
      - Fork creation (2+ simultaneous threats)
      - Open line counts for both players
      - Board occupancy (game phase proxy)
      - Max partial line through the action cell
    """
    row, col = action
    player, opponent = 1, -1

    # Simulate placing our piece
    sim = [list(r) for r in state]
    sim[row][col] = player
    sim_state = tuple(tuple(r) for r in sim)

    # Simulate opponent placing there (to detect blocking)
    sim_opp = [list(r) for r in state]
    sim_opp[row][col] = opponent
    sim_opp_state = tuple(tuple(r) for r in sim_opp)

    is_centre = (row == 1 and col == 1)
    is_corner = row in (0, 2) and col in (0, 2)

    my_threats_after = min(_count_threats(sim_state, player), 3)

    lines_through = [ln for ln in _WIN_LINES if (row, col) in ln]
    my_max = opp_max = 0
    for ln in lines_through:
        pc, _ = _line_score(state, ln, player)
        if pc >= 0: my_max = max(my_max, pc)
        opc, _ = _line_score(state, ln, opponent)
        if opc >= 0: opp_max = max(opp_max, opc)

    return [
        ("bias",               1),
        ("action_cell",        (row, col)),
        ("pos_centre",         int(is_centre)),
        ("pos_corner",         int(is_corner)),
        ("pos_edge",           int(not is_centre and not is_corner)),
        ("immediate_win",      int(_is_winner(sim_state, player))),
        ("blocks_win",         int(_is_winner(sim_opp_state, opponent))),
        ("my_threats_before",  min(_count_threats(state, player),   3)),
        ("opp_threats_before", min(_count_threats(state, opponent), 3)),
        ("my_threats_after",   my_threats_after),
        ("creates_fork",       int(my_threats_after >= 2)),
        ("my_open_lines",      _count_open_lines(state, player)),
        ("opp_open_lines",     _count_open_lines(state, opponent)),
        ("board_occupancy",    sum(state[r][c] != 0 for r in range(3) for c in range(3))),
        ("my_max_line",        my_max),
        ("opp_max_line",       opp_max),
    ]


# ── Reward function ───────────────────────────────────────────────────────────

def reward_function(state, action, next_state):
    """
    Reward signal for the transition state -> action -> next_state.

    Terminal rewards dominate; shaping terms guide early learning
    before terminal states are reached.
    """
    player, opponent = 1, -1

    if _is_winner(next_state, player):   return  1.0   # win
    if _is_winner(next_state, opponent): return -1.0   # loss
    if _is_draw(next_state):             return -0.05  # draw (missed win)

    reward = 0.0

    # Creating threats
    reward += 0.05 * (_count_threats(next_state, player) - _count_threats(state, player))

    # Blocking opponent threats
    opp_before = _count_threats(state, opponent)
    opp_after  = _count_threats(next_state, opponent)
    if opp_before > 0 and opp_after < opp_before:
        reward += 0.03 * (opp_before - opp_after)

    # Positional bonuses
    row, col = action
    if row == 1 and col == 1 and state[1][1] == 0:
        reward += 0.02
    elif (row, col) in ((0,0),(0,2),(2,0),(2,2)) and state[row][col] == 0:
        reward += 0.01

    return reward


# ── Exploration schedule ──────────────────────────────────────────────────────

def epsilon_schedule(episode):
    """
    Epsilon-greedy exploration rate for episode t.

    Starts at 0.9 (mostly explore), decays to 0.05 floor (mostly exploit).
        epsilon_t = max(0.05,  0.9 * 0.995^t)
    """
    return max(0.05, 0.9 * (0.995 ** episode))
