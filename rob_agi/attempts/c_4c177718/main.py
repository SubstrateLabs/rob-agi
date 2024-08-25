from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4c177718(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 15x15 input grid into a 9x15 output grid by:
    1. Finding the blue shape below the gray line
    2. Finding the rightmost non-zero shape above the gray line
    3. Placing these shapes in the new grid based on their original positions
    4. Centering the shapes horizontally and vertically
    5. Adjusting the grid to exactly 9 rows
    """
    def find_shape(grid: ColoredGrid, start_row: int, end_row: int) -> List[Tuple[int, int, int]]:
        shape = []
        visited = set()
        for r in range(start_row, end_row + 1):
            for c in range(15):
                if grid.values[r][c] != 0 and (r, c) not in visited:
                    color = grid.values[r][c]
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and start_row <= curr_r <= end_row and 0 <= curr_c < 15 and grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            shape.append((curr_r, curr_c, color))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                stack.append((curr_r + dr, curr_c + dc))
        return shape

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

    # Find blue shape below gray line
    blue_shape = find_shape(input_grid, 6, 14)

    # Find rightmost shape above gray line
    top_shape = None
    for c in range(14, -1, -1):
        for r in range(1, 5):
            if input_grid.values[r][c] != 0:
                top_shape = find_shape(input_grid, 1, 4)
                break
        if top_shape:
            break

    # Determine vertical arrangement
    blue_min_r, blue_max_r, blue_min_c, blue_max_c = get_shape_bounds(blue_shape)
    top_min_r, top_max_r, top_min_c, top_max_c = get_shape_bounds(top_shape)

    blue_height = blue_max_r - blue_min_r + 1
    top_height = top_max_r - top_min_r + 1
    total_height = blue_height + top_height + 1  # +1 for space between shapes

    # Calculate vertical positions
    if blue_min_r >= 11:  # Blue shape was in bottom third
        blue_top = 9 - blue_height
        top_top = blue_top - top_height - 1
    else:
        top_top = 0
        blue_top = top_height + 1

    # Center horizontally
    total_width = max(blue_max_c - blue_min_c, top_max_c - top_min_c) + 1
    left_col = (15 - total_width) // 2

    # Create new 9x15 grid
    new_grid = [[0 for _ in range(15)] for _ in range(9)]

    # Place shapes
    place_shape(new_grid, top_shape, top_top, left_col)
    place_shape(new_grid, blue_shape, blue_top, left_col)

    return ColoredGrid(values=new_grid)
