from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_4364c1c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by redistributing shapes towards the edges while maintaining vertical order:
    1. Identifies the background color as the most frequent color.
    2. Finds all distinct shapes (connected regions of non-background colors).
    3. Sorts shapes from top to bottom.
    4. Alternately moves shapes left and right:
       - Odd-indexed shapes: Move left as far as possible without overlapping.
       - Even-indexed shapes: Move right as far as possible without overlapping.
    5. Adjusts vertical positions to close gaps while maintaining order.
    6. Fine-tunes horizontal positions to ensure proper spacing.
    7. Places shapes on a new grid, avoiding overlaps and staying within bounds.
    """
    background_color = Counter([cell for row in input_grid.values for cell in row]).most_common(1)[0][0]
    shapes = find_shapes(input_grid, background_color)
    shapes.sort(key=lambda shape: min(cell[0] for cell in shape))

    new_grid = ColoredGrid(values=[[background_color for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])

    for i, shape in enumerate(shapes):
        color = input_grid.values[shape[0][0]][shape[0][1]]
        if i % 2 == 0:  # Move left
            new_shape = move_shape_left(shape, new_grid, background_color)
        else:  # Move right
            new_shape = move_shape_right(shape, new_grid, background_color)
        
        new_shape = adjust_vertical_position(new_shape, new_grid, background_color)
        place_shape(new_grid, new_shape, color)

    return new_grid

def move_shape_left(shape: List[Tuple[int, int]], grid: ColoredGrid, background_color: int) -> List[Tuple[int, int]]:
    min_col = min(c for _, c in shape)
    for col in range(min_col, -1, -1):
        new_shape = [(r, c - (min_col - col)) for r, c in shape]
        if all(0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == background_color for r, c in new_shape):
            return new_shape
    return shape

def move_shape_right(shape: List[Tuple[int, int]], grid: ColoredGrid, background_color: int) -> List[Tuple[int, int]]:
    max_col = max(c for _, c in shape)
    for col in range(max_col, grid.num_cols):
        new_shape = [(r, c + (col - max_col)) for r, c in shape]
        if all(0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid.values[r][c] == background_color for r, c in new_shape):
            return new_shape
    return shape

def adjust_vertical_position(shape: List[Tuple[int, int]], grid: ColoredGrid, background_color: int) -> List[Tuple[int, int]]:
    min_row = min(r for r, _ in shape)
    while min_row > 0:
        new_shape = [(r - 1, c) for r, c in shape]
        if all(grid.values[r][c] == background_color for r, c in new_shape):
            shape = new_shape
            min_row -= 1
        else:
            break
    return shape

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
