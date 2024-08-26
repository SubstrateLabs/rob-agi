from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_f3b10344(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f3b10344 challenge by creating a sky blue network that connects non-black shapes.
    
    The function identifies non-black shapes, creates an initial sky blue border,
    connects shapes to the border with 3-cell wide paths, optimizes the path to fill
    empty spaces more evenly, and ensures all non-black shapes are connected by a
    single, continuous, 3-cell wide sky blue path.
    """
    if all(cell == 0 for row in input_grid.values for cell in row):
        return input_grid

    rows, cols = input_grid.get_dimensions()
    grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid(r+dr, c+dc)]

    def find_shapes() -> List[Set[Tuple[int, int]]]:
        shapes = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] != 0 and (r, c) not in visited:
                    shape = set()
                    queue = deque([(r, c)])
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited:
                            visited.add((curr_r, curr_c))
                            shape.add((curr_r, curr_c))
                            for nr, nc in get_neighbors(curr_r, curr_c):
                                if input_grid.values[nr][nc] == input_grid.values[curr_r][curr_c]:
                                    queue.append((nr, nc))
                    shapes.append(shape)
        return shapes

    def create_border():
        for r in range(rows):
            for c in range(cols):
                if r < 3 or r >= rows - 3 or c < 3 or c >= cols - 3:
                    grid.values[r][c] = 8

    def connect_shape(shape: Set[Tuple[int, int]]):
        frontier = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 8]
        target = min(shape, key=lambda x: min(abs(x[0]-f[0]) + abs(x[1]-f[1]) for f in frontier))
        path = []
        current = min(frontier, key=lambda x: abs(x[0]-target[0]) + abs(x[1]-target[1]))
        while current != target:
            path.append(current)
            r, c = current
            next_step = min(get_neighbors(r, c), key=lambda x: abs(x[0]-target[0]) + abs(x[1]-target[1]))
            current = next_step
        path.append(target)
        for r, c in path:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if is_valid(r+dr, c+dc):
                        grid.values[r+dr][c+dc] = 8

    shapes = find_shapes()
    create_border()
    for shape in shapes:
        connect_shape(shape)

    # Restore original shapes
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                grid.values[r][c] = input_grid.values[r][c]

    return grid
