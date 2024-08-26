from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import deque

def solve_df8cc377(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying closed shapes, filling their interiors with patterns,
    and clearing the rest of the grid. The process involves:
    1. Detecting closed shapes in the grid using flood fill.
    2. Identifying the highest-numbered color present in the grid for filling.
    3. For shapes 5x5 or larger, filling the interior with a checkerboard pattern using
       the highest-numbered color and black (0).
    4. For 3x3 shapes, placing a single dot of the highest-numbered color at the center.
    5. For 5x3 or 3x5 shapes, placing two dots of the highest-numbered color symmetrically.
    6. For shapes smaller than 3x3, removing them entirely.
    7. Clearing all cells not part of any shape's boundary or interior.
    8. Reconstructing the grid with the modified shapes.
    """
    BLACK = 0

    def is_outline_color(color: int) -> bool:
        return color != BLACK and any(input_grid.get_cell(r, c) == color for r in range(input_grid.num_rows) for c in range(input_grid.num_cols))

    def get_highest_fill_color(grid: ColoredGrid) -> int:
        return max((color for color in range(9, 0, -1) if any(color in row for row in grid.values)), default=0)

    def find_shapes(grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        shapes = []
        visited = set()
        rows, cols = grid.get_dimensions()

        def flood_fill(r: int, c: int, color: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
            boundary = []
            interior = []
            queue = deque([(r, c)])
            while queue:
                curr_r, curr_c = queue.popleft()
                if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                    visited.add((curr_r, curr_c))
                    is_boundary = False
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid.get_cell(nr, nc) != color:
                                is_boundary = True
                            else:
                                queue.append((nr, nc))
                    if is_boundary:
                        boundary.append((curr_r, curr_c))
                    else:
                        interior.append((curr_r, curr_c))
            return boundary, interior

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) != BLACK:
                    boundary, interior = flood_fill(r, c, grid.get_cell(r, c))
                    if boundary:  # Only add shapes with a boundary
                        shapes.append((grid.get_cell(r, c), boundary, interior))

        return shapes

    def get_shape_dimensions(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        if not shape:
            return 0, 0, 0, 0
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1

    def apply_checkerboard(grid: ColoredGrid, shape: List[Tuple[int, int]], fill_color: int):
        if not shape:
            return
        min_r, min_c, _, _ = get_shape_dimensions(shape)
        for r, c in shape:
            if (r - min_r + c - min_c) % 2 == 0:
                grid.set_cell(r, c, fill_color)

    def fill_shape(grid: ColoredGrid, boundary: List[Tuple[int, int]], interior: List[Tuple[int, int]], shape_color: int, fill_color: int):
        min_r, min_c, height, width = get_shape_dimensions(boundary + interior)
        if height >= 5 and width >= 5:
            apply_checkerboard(grid, interior, fill_color)
        elif height == 3 and width == 3:
            center_r, center_c = min_r + 1, min_c + 1
            grid.set_cell(center_r, center_c, fill_color)
        elif (height == 5 and width == 3) or (height == 3 and width == 5):
            if height == 5:
                grid.set_cell(min_r + 1, min_c + 1, fill_color)
                grid.set_cell(min_r + 3, min_c + 1, fill_color)
            else:
                grid.set_cell(min_r + 1, min_c + 1, fill_color)
                grid.set_cell(min_r + 1, min_c + 3, fill_color)
        # For shapes smaller than 3x3, we don't fill them (they will be removed)

    new_grid = ColoredGrid(values=[[BLACK for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    shapes = find_shapes(input_grid)
    fill_color = get_highest_fill_color(input_grid)

    for shape_color, boundary, interior in shapes:
        if len(boundary) + len(interior) >= 9:  # Only process shapes 3x3 or larger
            # Add boundary to new_grid
            for r, c in boundary:
                new_grid.set_cell(r, c, shape_color)
            fill_shape(new_grid, boundary, interior, shape_color, fill_color)

    return new_grid
