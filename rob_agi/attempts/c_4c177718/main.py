from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_4c177718(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 15x15 input grid into a 9x15 output grid by:
    1. Finding the blue shape below the gray line
    2. Finding the rightmost non-zero shape above the gray line
    3. Placing these shapes in the new grid based on their original positions
    4. Centering the shapes horizontally
    5. Trimming or padding the grid to exactly 9 rows
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

    def center_shape(shape: List[Tuple[int, int, int]], center_column: int) -> List[Tuple[int, int, int]]:
        min_c = min(c for _, c, _ in shape)
        max_c = max(c for _, c, _ in shape)
        shift = center_column - (min_c + max_c) // 2
        return [(r, c + shift, color) for r, c, color in shape]

    def place_shape(grid: List[List[int]], shape: List[Tuple[int, int, int]], top_row: int):
        for r, c, color in shape:
            grid[top_row + r - min(r for r, _, _ in shape)][c] = color

    # Find blue shape below gray line
    blue_shape = find_shape(input_grid, 6, 14)

    # Find rightmost shape above gray line
    top_shape = None
    for c in range(14, -1, -1):
        for r in range(1, 4):
            if input_grid.values[r][c] != 0:
                top_shape = find_shape(input_grid, 1, 4)
                break
        if top_shape:
            break

    # Create new 9x15 grid
    new_grid = [[0 for _ in range(15)] for _ in range(9)]

    # Determine order and place shapes
    if top_shape[0][0] == 1:  # Top shape was in the first row
        place_shape(new_grid, center_shape(top_shape, 7), 1)
        place_shape(new_grid, center_shape(blue_shape, 7), 5)
    else:
        place_shape(new_grid, center_shape(blue_shape, 7), 1)
        place_shape(new_grid, center_shape(top_shape, 7), 5)

    # Trim or pad the grid to exactly 9 rows
    while len(new_grid) > 9 and all(cell == 0 for cell in new_grid[-1]):
        new_grid.pop()
    while len(new_grid) < 9:
        new_grid.append([0] * 15)

    return ColoredGrid(values=new_grid)
