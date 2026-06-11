import heapq
from typing import Callable
from src.puzzle import get_neighbors, is_goal

def astar(start, heuristic: Callable, evaluate: Callable):
    frontier = []
    heapq.heappush(frontier, (0, 0, start))

    visited = {start: 0}
    expansions = 0

    while frontier:
        f, g, state = heapq.heappop(frontier)

        if is_goal(state):
            return {
                "cost": g,
                "expanded": expansions
            }

        expansions += 1

        for neighbor in get_neighbors(state):
            new_g = g + 1

            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                h = heuristic(neighbor)
                new_f = evaluate(new_g, h)
                heapq.heappush(frontier, (new_f, new_g, neighbor))

    return None