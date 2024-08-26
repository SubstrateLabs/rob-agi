from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_4364c1c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving shapes with alternating left-right movements:
    1. Identifies the background color as the most frequent color.
    2. Finds all distinct shapes (connected regions of non-background colors).
    3. Sorts shapes from top to bottom.
    4. Moves shapes in pairs:
       - First shape of each pair: Move left with increasing amounts (1, 2, 3, ...)
       - Second shape of each pair: Move right and down with increasing amounts (1, 2, 3, ...)
    5. Applies movements while keeping shapes within grid bounds and preventing overlaps.
    """
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]
    shapes = find_shapes(input_grid, background_color)
    shapes.sort(key=lambda shape: min(cell[0] for cell in shape))

    new_grid = ColoredGrid(values=[[background_color for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])

    for i in range(0, len(shapes), 2):
        movement = (i // 2) + 1
        
        # Move first shape left
        if i < len(shapes):
            shape1 = shapes[i]
            new_shape1 = move_shape(shape1, -movement, 0, new_grid.num_rows, new_grid.num_cols)
            color1 = input_grid.values[shape1[0][0]][shape1[0][1]]
            new_shape1 = adjust_shape_position(new_grid, new_shape1, background_color)
            place_shape(new_grid, new_shape1, color1)
        
        # Move second shape right and down
        if i + 1 < len(shapes):
            shape2 = shapes[i + 1]
            new_shape2 = move_shape(shape2, movement, movement, new_grid.num_rows, new_grid.num_cols)
            color2 = input_grid.values[shape2[0][0]][shape2[0][1]]
            new_shape2 = adjust_shape_position(new_grid, new_shape2, background_color)
            place_shape(new_grid, new_shape2, color2)

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

def adjust_shape_position(grid: ColoredGrid, shape: List[Tuple[int, int]], background_color: int) -> List[Tuple[int, int]]:
    if shape is None:
        return None
    
    while any(r < 0 for r, _ in shape):
        shape = [(r + 1, c) for r, c in shape]  # Move shape down if it's above the top edge
    
    while any(r >= grid.num_rows or c >= grid.num_cols or grid.values[r][c] != background_color for r, c in shape):
        shape = [(r - 1, c) for r, c in shape]  # Move shape up if it overlaps or is out of bounds
        if any(r < 0 for r, _ in shape):
            return None  # Cannot place the shape without overlap
    
    return shape

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], color: int):
    if shape is not None:
        for r, c in shape:
            grid.values[r][c] = color
