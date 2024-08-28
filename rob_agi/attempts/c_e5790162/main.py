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
    5. If edge extension is not possible, backtracks to find an efficient extension point.
    6. Doesn't overwrite existing colored squares.
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

    def connect_targets(colored_squares: List[Tuple[int, int, int]], grid: List[List[int]]) -> List[Tuple[int, int]]:
        start = next((sq for sq in colored_squares if sq[2] == 3), colored_squares[0])
        magenta_targets = [sq[:2] for sq in colored_squares if sq[2] == 6]
        sky_targets = [sq[:2] for sq in colored_squares if sq[2] == 8]
        
        current = start[:2]
        path = [current]
        for target in magenta_targets + sky_targets:
            segment = a_star(current, [target], grid)
            path.extend(segment[1:])  # Don't duplicate the start of each segment
            for r, c in segment[1:-1]:  # Don't overwrite the target
                if grid[r][c] == 0:
                    grid[r][c] = 3
            current = target
        return path

    def extend_to_edge(grid: List[List[int]], path: List[Tuple[int, int]]) -> None:
        rows, cols = len(grid), len(grid[0])
        
        def try_extend(start: Tuple[int, int], direction: Tuple[int, int]) -> bool:
            r, c = start
            dr, dc = direction
            for _ in range(3):
                r, c = r + dr, c + dc
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    return True  # Reached the edge
                if grid[r][c] != 0:
                    return False  # Hit a non-empty cell
                grid[r][c] = 3
            return False

        # Try to extend from the last point
        last_point = path[-1]
        second_last_point = path[-2] if len(path) > 1 else None
        if second_last_point:
            direction = (last_point[0] - second_last_point[0], last_point[1] - second_last_point[1])
            if try_extend(last_point, direction):
                return

        # Backtrack and try to extend
        for i in range(len(path) - 2, -1, -1):
            current = path[i]
            next_point = path[i + 1]
            direction = (next_point[0] - current[0], next_point[1] - current[1])
            if try_extend(current, direction):
                return

    output_grid = [row[:] for row in input_grid.values]
    colored_squares = find_colored_squares()
    path = connect_targets(colored_squares, output_grid)
    extend_to_edge(output_grid, path)

    return ColoredGrid(values=output_grid)
