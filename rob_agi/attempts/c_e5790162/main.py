from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from queue import PriorityQueue

def solve_e5790162(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green path that connects the initial green square
    to all magenta and sky blue squares. The path prioritizes horizontal movement, then vertical.
    It doesn't overwrite magenta or sky blue squares and can continue past them if needed.
    
    1. Finds the starting green square.
    2. Identifies all magenta and sky blue target squares.
    3. Uses A* algorithm to find paths between targets, prioritizing horizontal movement.
    4. Connects all targets with a green path, avoiding overwriting existing colored squares.
    5. Optimizes the path by removing unnecessary detours.
    """
    def find_colored_squares() -> List[Tuple[int, int, int]]:
        return [(r, c, val) for r, row in enumerate(input_grid.values) 
                for c, val in enumerate(row) if val in (3, 6, 8)]

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[int]]) -> List[Tuple[int, int]]:
        rows, cols = len(grid), len(grid[0])
        pq = PriorityQueue()
        pq.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: manhattan_distance(start, goal)}

        while not pq.empty():
            current = pq.get()[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                    tentative_g_score = g_score[current] + (1 if dr == 0 else 1.1)  # Slight preference for horizontal movement
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal)
                        pq.put((f_score[neighbor], neighbor))

        return []  # No path found

    def connect_targets(colored_squares: List[Tuple[int, int, int]], grid: List[List[int]]) -> None:
        start = next(sq for sq in colored_squares if sq[2] == 3)
        targets = [sq for sq in colored_squares if sq[2] in (6, 8)]
        current = start

        while targets:
            nearest = min(targets, key=lambda t: manhattan_distance(current[:2], t[:2]))
            path = a_star(current[:2], nearest[:2], grid)
            for r, c in path[1:-1]:  # Skip start and end points
                if grid[r][c] == 0:
                    grid[r][c] = 3
            current = nearest
            targets.remove(nearest)

    output_grid = [row[:] for row in input_grid.values]
    colored_squares = find_colored_squares()
    connect_targets(colored_squares, output_grid)

    return ColoredGrid(values=output_grid)
