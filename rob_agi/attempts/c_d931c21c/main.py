from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_d931c21c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies closed shapes formed by blue (1) cells, including those touching grid edges.
    2. For each identified closed shape:
       - Transforms adjacent black (0) cells outside the shape to red (2).
       - Transforms all black cells inside the shape to green (3).
    3. Blue cells remain unchanged.
    4. If no closed shapes are found, the original grid is returned.

    The transformation is applied only once, not iteratively.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    visited = set()

    def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Only orthogonal neighbors
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                neighbors.append((new_row, new_col))
        return neighbors

    def trace_shape(start_row: int, start_col: int) -> Set[Tuple[int, int]]:
        shape = set()
        queue = deque([(start_row, start_col)])
        while queue:
            r, c = queue.popleft()
            if (r, c) not in shape:
                shape.add((r, c))
                for nr, nc in get_neighbors(r, c):
                    if new_grid.get_cell(nr, nc) == 1 and (nr, nc) not in shape:  # Blue
                        queue.append((nr, nc))
        return shape

    def is_closed_shape(shape: Set[Tuple[int, int]]) -> bool:
        outside = set()
        inside = set()
        for r, c in shape:
            for nr, nc in get_neighbors(r, c):
                if (nr, nc) not in shape:
                    if nr == 0 or nr == rows - 1 or nc == 0 or nc == cols - 1:
                        outside.add((nr, nc))
                    elif new_grid.get_cell(nr, nc) == 0:  # Black
                        inside.add((nr, nc))
        return len(outside) == 0 or len(inside) == 0

    def flood_fill(start_row: int, start_col: int, shape: Set[Tuple[int, int]], color: int) -> None:
        queue = deque([(start_row, start_col)])
        while queue:
            r, c = queue.popleft()
            if (r, c) not in shape and new_grid.get_cell(r, c) == 0:  # Black
                new_grid.set_cell(r, c, color)
                for nr, nc in get_neighbors(r, c):
                    if (nr, nc) not in shape:
                        queue.append((nr, nc))

    changes_made = False
    for row in range(rows):
        for col in range(cols):
            if new_grid.get_cell(row, col) == 1 and (row, col) not in visited:  # Blue
                shape = trace_shape(row, col)
                if is_closed_shape(shape):
                    visited.update(shape)
                    for r, c in shape:
                        for nr, nc in get_neighbors(r, c):
                            if (nr, nc) not in shape:
                                if new_grid.get_cell(nr, nc) == 0:  # Black
                                    new_grid.set_cell(nr, nc, 2)  # Red
                                    changes_made = True
                                elif new_grid.get_cell(nr, nc) == 2:  # Red
                                    flood_fill(nr, nc, shape, 3)  # Green
                                    changes_made = True

    return new_grid if changes_made else input_grid
