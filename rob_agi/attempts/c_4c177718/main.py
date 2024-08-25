from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4c177718(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 15x15 input grid into a 9x15 output grid by:
    1. Identifying the gray line (always in row 5)
    2. Finding the rightmost non-blue, non-red shape above the gray line (Shape A)
    3. Finding the single shape below the gray line (Shape B)
    4. If Shape A is not blue, placing A at the top and B at the bottom
       If Shape A is blue, placing B at the top and A at the bottom
    5. Centering both shapes horizontally and vertically in their respective halves
    6. Creating a new 9x15 grid with the arranged shapes
    """
    def find_shapes(grid: ColoredGrid, start_row: int, end_row: int) -> List[List[Tuple[int, int, int]]]:
        shapes = []
        visited = set()
        for r in range(start_row, end_row + 1):
            for c in range(15):
                if grid.values[r][c] != 0 and (r, c) not in visited:
                    color = grid.values[r][c]
                    shape = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and start_row <= curr_r <= end_row and 0 <= curr_c < 15 and grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            shape.append((curr_r, curr_c, color))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                stack.append((curr_r + dr, curr_c + dc))
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

    # Find shapes above gray line
    top_shapes = find_shapes(input_grid, 1, 4)

    # Find shape below gray line
    bottom_shapes = find_shapes(input_grid, 6, 14)
    bottom_shape = bottom_shapes[0] if bottom_shapes else []

    # Select Shape A (rightmost non-blue, non-red shape above gray line)
    shape_a = None
    for shape in reversed(top_shapes):
        if shape[0][2] not in [1, 2]:  # Not blue or red
            shape_a = shape
            break

    # Determine arrangement
    if shape_a[0][2] != 1:  # If Shape A is not blue
        top_shape, bottom_shape = shape_a, bottom_shape
    else:
        top_shape, bottom_shape = bottom_shape, shape_a

    # Create new 9x15 grid
    new_grid = [[0 for _ in range(15)] for _ in range(9)]

    # Place top shape
    _, _, top_min_c, top_max_c = get_shape_bounds(top_shape)
    top_width = top_max_c - top_min_c + 1
    top_left_col = (15 - top_width) // 2
    place_shape(new_grid, top_shape, 0, top_left_col)

    # Place bottom shape
    _, _, bottom_min_c, bottom_max_c = get_shape_bounds(bottom_shape)
    bottom_width = bottom_max_c - bottom_min_c + 1
    bottom_left_col = (15 - bottom_width) // 2
    place_shape(new_grid, bottom_shape, 5, bottom_left_col)

    return ColoredGrid(values=new_grid)
