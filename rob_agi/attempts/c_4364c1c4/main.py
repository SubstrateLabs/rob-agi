from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_4364c1c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving shapes based on their vertical position:
    1. Identifies the background color as the most frequent color.
    2. Finds all distinct shapes (connected regions of non-background colors).
    3. Sorts shapes from top to bottom.
    4. Moves shapes:
       - Topmost shape: 1 cell left and 1 cell up
       - Bottommost shape: 3 cells right and 1 cell down
       - Odd-numbered shapes from top: 1 cell left
       - Even-numbered shapes from top: 3 cells right
    5. Applies movements while keeping shapes within grid bounds.
    """
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]
    shapes = find_shapes(input_grid, background_color)
    shapes.sort(key=lambda shape: min(cell[0] for cell in shape))

    new_grid = ColoredGrid(values=[[background_color for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])

    for i, shape in enumerate(shapes):
        if i == 0:  # Topmost shape
            dx, dy = -1, -1
        elif i == len(shapes) - 1:  # Bottommost shape
            dx, dy = 3, 1
        elif i % 2 == 1:  # Odd-numbered shapes
            dx, dy = -1, 0
        else:  # Even-numbered shapes
            dx, dy = 3, 0

        new_shape = move_shape(shape, dx, dy, new_grid.num_rows, new_grid.num_cols)
        color = input_grid.values[shape[0][0]][shape[0][1]]
        place_shape(new_grid, new_shape, color)

    return new_grid

def find_shapes(grid: ColoredGrid, background_color: int) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()

    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.values[r][c] != background_color and (r, c) not in visited:
                shape = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == grid.values[r][c]:
                        visited.add((curr_r, curr_c))
                        shape.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            new_r, new_c = curr_r + dr, curr_c + dc
                            if 0 <= new_r < grid.num_rows and 0 <= new_c < grid.num_cols:
                                stack.append((new_r, new_c))
                shapes.append(shape)
    return shapes

def move_shape(shape: List[Tuple[int, int]], dx: int, dy: int, max_row: int, max_col: int) -> List[Tuple[int, int]]:
    new_shape = [(r + dy, c + dx) for r, c in shape]
    min_r = min(r for r, _ in new_shape)
    min_c = min(c for _, c in new_shape)
    max_r = max(r for r, _ in new_shape)
    max_c = max(c for _, c in new_shape)

    if min_r < 0:
        new_shape = [(r - min_r, c) for r, c in new_shape]
    elif max_r >= max_row:
        new_shape = [(r - (max_r - max_row + 1), c) for r, c in new_shape]

    if min_c < 0:
        new_shape = [(r, c - min_c) for r, c in new_shape]
    elif max_c >= max_col:
        new_shape = [(r, c - (max_c - max_col + 1)) for r, c in new_shape]

    return new_shape

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], color: int):
    for r, c in shape:
        grid.values[r][c] = color
