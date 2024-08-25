from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_df8cc377(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying closed shapes, filling their interiors with patterns,
    and clearing the rest of the grid. The process involves:
    1. Identifying closed shapes in the grid.
    2. For shapes 4x4 or larger, filling the interior with a checkerboard pattern using
       the highest-numbered color found inside the shape and black (0).
    3. For 3x3 shapes, placing a single dot of the highest-numbered color at the center.
    4. For 5x3 or 3x5 shapes, placing two dots of the highest-numbered color symmetrically.
    5. For 2x2 shapes and smaller, preserving them as they are.
    6. Clearing all cells not part of any shape's boundary or interior.
    7. Reconstructing the grid with the modified shapes.
    """
    def find_shapes(grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        shapes = []
        visited = set()
        rows, cols = grid.get_dimensions()

        def flood_fill(r: int, c: int, color: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
            boundary = []
            interior = []
            stack = [(r, c)]
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                    visited.add((curr_r, curr_c))
                    is_boundary = False
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid.get_cell(nr, nc) != color:
                                is_boundary = True
                            else:
                                stack.append((nr, nc))
                    if is_boundary:
                        boundary.append((curr_r, curr_c))
                    else:
                        interior.append((curr_r, curr_c))
            return boundary, interior

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) != 0:
                    boundary, interior = flood_fill(r, c, grid.get_cell(r, c))
                    shapes.append((grid.get_cell(r, c), boundary, interior))

        return shapes

    def get_shape_dimensions(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1

    def get_highest_color(grid: ColoredGrid, interior: List[Tuple[int, int]]) -> int:
        return max((grid.get_cell(r, c) for r, c in interior), default=0)

    def fill_shape(grid: ColoredGrid, boundary: List[Tuple[int, int]], interior: List[Tuple[int, int]], shape_color: int, fill_color: int):
        min_r, min_c, height, width = get_shape_dimensions(boundary + interior)
        if height >= 4 and width >= 4:
            for r, c in interior:
                grid.set_cell(r, c, fill_color if (r + c) % 2 == 0 else 0)
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
        # For 2x2 and smaller shapes, we don't modify them

    new_grid = ColoredGrid(values=[[0 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
    shapes = find_shapes(input_grid)

    for shape_color, boundary, interior in shapes:
        # Add boundary to new_grid
        for r, c in boundary:
            new_grid.set_cell(r, c, shape_color)
        
        if len(boundary) + len(interior) > 4:  # Only process shapes larger than 2x2
            fill_color = get_highest_color(input_grid, interior)
            fill_shape(new_grid, boundary, interior, shape_color, fill_color)
        else:
            # For 2x2 and smaller shapes, copy them as they are
            for r, c in interior:
                new_grid.set_cell(r, c, input_grid.get_cell(r, c))

    return new_grid
