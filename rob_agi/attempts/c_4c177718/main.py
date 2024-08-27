from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4c177718(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 15x15 input grid into a 9x15 output grid by:
    1. Identifying all shapes in the input grid
    2. Discarding the red shape (color code 2)
    3. Selecting the shape with the lowest color code from above the gray line
    4. Selecting the shape with the highest color code from below the gray line
    5. Placing the shape from above the line in the top half of the output grid
    6. Placing the shape from below the line in the bottom half of the output grid
    7. Centering both shapes horizontally and vertically in their respective halves
    """
    def find_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int, int]]]:
        shapes = []
        visited = set()
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] not in [0, 5] and (r, c) not in visited:
                    color = grid.values[r][c]
                    shape = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and 0 <= curr_r < rows and 0 <= curr_c < cols and grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            shape.append((curr_r, curr_c, color))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    stack.append((nr, nc))
                    shapes.append(shape)
        return shapes

    def get_shape_bounds(shape: List[Tuple[int, int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(r for r, _, _ in shape)
        max_r = max(r for r, _, _ in shape)
        min_c = min(c for _, c, _ in shape)
        max_c = max(c for _, c, _ in shape)
        return min_r, max_r, min_c, max_c

    def place_shape(grid: List[List[int]], shape: List[Tuple[int, int, int]], top_row: int, left_col: int):
        min_r, _, min_c, _ = get_shape_bounds(shape)
        for r, c, color in shape:
            grid[top_row + r - min_r][left_col + c - min_c] = color

    # Find all shapes
    shapes = find_shapes(input_grid)

    # Find the gray line
    gray_line_row = next(r for r, row in enumerate(input_grid.values) if 5 in row)

    # Filter shapes and select the appropriate ones
    shapes_above = [shape for shape in shapes if shape[0][0] < gray_line_row and shape[0][2] != 2]
    shapes_below = [shape for shape in shapes if shape[0][0] > gray_line_row]

    top_shape = min(shapes_above, key=lambda x: x[0][2])
    bottom_shape = max(shapes_below, key=lambda x: x[0][2])

    # Create new 9x15 grid
    new_grid = [[0 for _ in range(15)] for _ in range(9)]

    # Place top shape
    top_min_r, top_max_r, top_min_c, top_max_c = get_shape_bounds(top_shape)
    top_height = top_max_r - top_min_r + 1
    top_width = top_max_c - top_min_c + 1
    top_left_col = (15 - top_width) // 2
    top_top_row = (4 - top_height) // 2
    place_shape(new_grid, top_shape, top_top_row, top_left_col)

    # Place bottom shape
    bottom_min_r, bottom_max_r, bottom_min_c, bottom_max_c = get_shape_bounds(bottom_shape)
    bottom_height = bottom_max_r - bottom_min_r + 1
    bottom_width = bottom_max_c - bottom_min_c + 1
    bottom_left_col = (15 - bottom_width) // 2
    bottom_top_row = 5 + (4 - bottom_height) // 2
    place_shape(new_grid, bottom_shape, bottom_top_row, bottom_left_col)

    return ColoredGrid(values=new_grid)
