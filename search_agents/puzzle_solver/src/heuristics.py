GOAL_POS = {
    1:(0,0), 2:(0,1), 3:(0,2),
    4:(1,0), 5:(1,1), 6:(1,2),
    7:(2,0), 8:(2,1)
}

def heuristic(state):
    """
    Linear conflict + Manhattan distance.

    Manhattan distance is admissible. Linear conflict adds 2 for every pair
    of tiles in the same row/col that are both headed to that row/col but
    in the wrong order — still admissible, but much tighter.
    """
    total = 0

    # ── Manhattan distance 
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        goal_r, goal_c = GOAL_POS[tile]
        cur_r, cur_c = idx // 3, idx % 3
        total += abs(cur_r - goal_r) + abs(cur_c - goal_c)

    # ── Linear conflict (rows) 
    for row in range(3):
        tiles_in_row = []
        for col in range(3):
            tile = state[row * 3 + col]
            if tile != 0 and GOAL_POS[tile][0] == row:
                tiles_in_row.append((col, GOAL_POS[tile][1]))
        # count inversions
        for i in range(len(tiles_in_row)):
            for j in range(i + 1, len(tiles_in_row)):
                if tiles_in_row[i][1] > tiles_in_row[j][1]:
                    total += 2

    # ── Linear conflict (cols) 
    for col in range(3):
        tiles_in_col = []
        for row in range(3):
            tile = state[row * 3 + col]
            if tile != 0 and GOAL_POS[tile][1] == col:
                tiles_in_col.append((row, GOAL_POS[tile][0]))
        for i in range(len(tiles_in_col)):
            for j in range(i + 1, len(tiles_in_col)):
                if tiles_in_col[i][1] > tiles_in_col[j][1]:
                    total += 2

    return total


def evaluate(g, h):
    return g + h