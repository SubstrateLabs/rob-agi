from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_d931c21c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Identifies closed or nearly closed shapes formed by blue (1) cells.
    2. For each identified shape:
       - Transforms adjacent black (0) cells outside the shape to red (2).
       - Transforms black cells inside the shape to green (3).
    3. Blue cells and other colored cells remain unchanged.
    4. If no closed or nearly closed shapes are found, the original grid is returned.

    The transformation is applied only once, not iteratively.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    visited = set()
    changes_made = False

    def get_neighbors(row: int, col: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    neighbors.append((new_row, new_col))
        return neighbors

    def trace_shape(start_row: int, start_col: int) -> Set[Tuple[int, int]]:
        shape = set()
        stack = [(start_row, start_col)]
        open_edges = 0

        while stack:
            r, c = stack.pop()
            if (r, c) not in shape:
                shape.add((r, c))
                neighbors = get_neighbors(r, c)
                for nr, nc in neighbors:
                    if new_grid.get_cell(nr, nc) == 1:  # Blue
                        if (nr, nc) not in shape:
                            stack.append((nr, nc))
                    else:
                        open_edges += 1

        return shape if open_edges <= 2 else set()

    def flood_fill(start_row: int, start_col: int, shape: Set[Tuple[int, int]]) -> None:
        queue = [(start_row, start_col)]
        while queue:
            r, c = queue.pop(0)
            if new_grid.get_cell(r, c) == 0:  # Black
                new_grid.set_cell(r, c, 3)  # Green
                nonlocal changes_made
                changes_made = True
                neighbors = get_neighbors(r, c)
                for nr, nc in neighbors:
                    if (nr, nc) not in shape and new_grid.get_cell(nr, nc) == 0:
                        queue.append((nr, nc))

    for row in range(rows):
        for col in range(cols):
            if new_grid.get_cell(row, col) == 1 and (row, col) not in visited:  # Blue
                shape = trace_shape(row, col)
                if shape:
                    visited.update(shape)
                    for r, c in shape:
                        neighbors = get_neighbors(r, c)
                        for nr, nc in neighbors:
                            if (nr, nc) not in shape and new_grid.get_cell(nr, nc) == 0:
                                new_grid.set_cell(nr, nc, 2)  # Red
                                changes_made = True
                    
                    # Find a starting point inside the shape for flood fill
                    for r, c in shape:
                        inner_neighbors = get_neighbors(r, c)
                        for nr, nc in inner_neighbors:
                            if (nr, nc) not in shape and new_grid.get_cell(nr, nc) == 0:
                                flood_fill(nr, nc, shape)
                                break
                        if changes_made:
                            break

    return new_grid if changes_made else input_grid
