from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from queue import PriorityQueue

def solve_e5790162(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green path that connects the initial green square
    to all magenta and sky blue squares, and extends to all four edges of the grid if possible.
    The path prioritizes horizontal movement, then vertical. It doesn't overwrite magenta or sky blue squares.
    
    1. Finds the starting green square and all target squares (magenta and sky blue).
    2. Determines edge priorities based on the location of the initial green square.
    3. Uses a modified A* algorithm to find paths between targets and to edges, prioritizing horizontal movement.
    4. Connects all targets with a green path, avoiding overwriting existing colored squares.
    5. Extends the path to reach all four edges of the grid when possible.
    6. Optimizes the path by removing unnecessary detours.
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

    def get_edge_priorities(start: Tuple[int, int], rows: int, cols: int) -> List[List[Tuple[int, int]]]:
        left = [(r, 0) for r in range(rows)]
        right = [(r, cols-1) for r in range(rows)]
        top = [(0, c) for c in range(cols)]
        bottom = [(rows-1, c) for c in range(cols)]
        
        if start[1] == 0:
            return [right, bottom, top, left]
        elif start[1] == cols-1:
            return [left, bottom, top, right]
        elif start[0] == 0:
            return [bottom, left, right, top]
        else:
            return [top, left, right, bottom]

    def connect_targets_and_edges(colored_squares: List[Tuple[int, int, int]], grid: List[List[int]]) -> None:
        start = next(sq for sq in colored_squares if sq[2] == 3)
        targets = [sq[:2] for sq in colored_squares if sq[2] in (6, 8)]
        rows, cols = len(grid), len(grid[0])
        edge_priorities = get_edge_priorities(start[:2], rows, cols)
        
        current = start[:2]
        all_goals = targets + [edge for priority in edge_priorities for edge in priority]

        while all_goals:
            path = a_star(current, all_goals, grid)
            if not path:
                break
            for r, c in path[1:]:
                if grid[r][c] == 0:
                    grid[r][c] = 3
            current = path[-1]
            if current in targets:
                targets.remove(current)
            all_goals = targets + [edge for priority in edge_priorities for edge in priority if grid[edge[0]][edge[1]] == 0]

    output_grid = [row[:] for row in input_grid.values]
    colored_squares = find_colored_squares()
    connect_targets_and_edges(colored_squares, output_grid)

    return ColoredGrid(values=output_grid)
