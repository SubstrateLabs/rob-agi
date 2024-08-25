from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_df8cc377(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying closed shapes, filling their interiors with patterns,
    and clearing the rest of the grid. The process involves:
    1. Identifying closed shapes in the grid.
    2. For each shape larger than 3x3, filling its interior with a checkerboard pattern using
       the highest-numbered color found inside the shape and black (0).
    3. For shapes 3x3 or smaller, placing dot(s) of the highest-numbered color at the center.
    4. Clearing all cells not part of any shape's boundary.
    5. Reconstructing the grid with the modified shapes.
    """
    def find_shapes(grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]]]]:
        shapes = []
        visited = set()
        rows, cols = grid.get_dimensions()

        def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int]]:
            shape = []
            stack = [(r, c)]
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                    visited.add((curr_r, curr_c))
                    shape.append((curr_r, curr_c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return shape

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) != 0:
                    shape = flood_fill(r, c, grid.get_cell(r, c))
                    shapes.append((grid.get_cell(r, c), shape))

        return shapes

    def get_shape_dimensions(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        return min_r, min_c, max_r - min_r + 1, max_c - min_c + 1

    def get_highest_color(grid: ColoredGrid, shape: List[Tuple[int, int]], shape_color: int) -> int:
        return max((grid.get_cell(r, c) for r, c in shape if grid.get_cell(r, c) != shape_color), default=0)

    def fill_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], shape_color: int, fill_color: int):
        min_r, min_c, height, width = get_shape_dimensions(shape)
        if height > 3 and width > 3:
            for r in range(min_r + 1, min_r + height - 1):
                for c in range(min_c + 1, min_c + width - 1):
                    if (r, c) in shape:
                        grid.set_cell(r, c, fill_color if (r + c) % 2 == 0 else 0)
        elif height == 3 and width == 3:
            center_r, center_c = min_r + 1, min_c + 1
            grid.set_cell(center_r, center_c, fill_color)
        elif height == 2 and width == 2:
            for r in range(min_r, min_r + 2):
                for c in range(min_c, min_c + 2):
                    grid.set_cell(r, c, fill_color)

    new_grid = input_grid.deep_copy()
    shapes = find_shapes(new_grid)

    for shape_color, shape in shapes:
        fill_color = get_highest_color(new_grid, shape, shape_color)
        fill_shape(new_grid, shape, shape_color, fill_color)

    rows, cols = new_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if new_grid.get_cell(r, c) not in [shape_color for shape_color, _ in shapes]:
                new_grid.set_cell(r, c, 0)

    return new_grid
