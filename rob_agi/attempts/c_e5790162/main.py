from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from queue import PriorityQueue

def solve_e5790162(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green path that connects the initial green square
    to all magenta and sky blue squares, and extends to an edge of the grid if efficient.
    
    1. Locates the starting green square and all target squares (magenta and sky blue).
    2. Connects targets using a pathfinding algorithm, prioritizing magenta squares before sky blue.
    3. Creates a single, continuous path without branches.
    4. Extends the path to an edge if it can do so within 3 steps in the last segment's direction.
    5. Doesn't overwrite existing colored squares.
    """
    def find_colored_squares() -> List[Tuple[int, int, int]]:
        return [(r, c, val) for r, row in enumerate(input_grid.values) 
                for c, val in enumerate(row) if val in (3, 6, 8)]

    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def a_star(start: Tuple[int, int], goals: List[Tuple[int, int]], grid: List[List[int]]) -> List[Tuple[int, int]]:
        rows, cols = len(grid), len(grid[0])
        pq = PriorityQueue()
        pq.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: min(manhattan_distance(start, goal) for goal in goals)}

        while not pq.empty():
            current = pq.get()[1]

            if current in goals:
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
                        f_score[neighbor] = g_score[neighbor] + min(manhattan_distance(neighbor, goal) for goal in goals)
                        pq.put((f_score[neighbor], neighbor))

        return []  # No path found

    def connect_targets(colored_squares: List[Tuple[int, int, int]], grid: List[List[int]]) -> None:
        start = next((sq for sq in colored_squares if sq[2] == 3), colored_squares[0])
        magenta_targets = [sq[:2] for sq in colored_squares if sq[2] == 6]
        sky_targets = [sq[:2] for sq in colored_squares if sq[2] == 8]
        
        current = start[:2]
        for target in magenta_targets + sky_targets:
            path = a_star(current, [target], grid)
            for r, c in path[1:-1]:  # Don't overwrite the target
                if grid[r][c] == 0:
                    grid[r][c] = 3
            current = target

    def extend_to_edge(grid: List[List[int]]) -> None:
        rows, cols = len(grid), len(grid[0])
        green_squares = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 3]
        
        if not green_squares:
            return

        last_green = green_squares[-1]
        second_last_green = green_squares[-2] if len(green_squares) > 1 else None

        if second_last_green:
            dr = last_green[0] - second_last_green[0]
            dc = last_green[1] - second_last_green[1]
        else:
            dr, dc = 0, 1  # Default to horizontal if only one green square

        for i in range(1, 4):
            r, c = last_green[0] + i*dr, last_green[1] + i*dc
            if r < 0 or r >= rows or c < 0 or c >= cols:
                break
            if grid[r][c] != 0:
                return
            grid[r][c] = 3

    def get_line(r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
        line = []
        if r1 == r2:  # Horizontal line
            for c in range(min(c1, c2), max(c1, c2) + 1):
                line.append((r1, c))
        elif c1 == c2:  # Vertical line
            for r in range(min(r1, r2), max(r1, r2) + 1):
                line.append((r, c1))
        return line

    output_grid = [row[:] for row in input_grid.values]
    colored_squares = find_colored_squares()
    connect_targets(colored_squares, output_grid)
    extend_to_edge(output_grid)

    return ColoredGrid(values=output_grid)
