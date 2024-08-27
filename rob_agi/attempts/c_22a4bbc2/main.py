from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_22a4bbc2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing qualifying rectangles to red.
    
    A qualifying rectangle is a contiguous area of blue (1) or sky blue (8),
    with dimensions 2x2 or larger, including 3x1, 1x3, 3x2, 2x3, 3x3, 4x1, or 1x4.
    The function identifies all such rectangles and changes them to red (color 2).
    Larger qualifying shapes take precedence over smaller ones.
    Overlapping or adjacent qualifying rectangles are merged into larger red areas.
    All changes are applied simultaneously to the input grid.
    Non-qualifying blue or sky blue areas remain unchanged.
    Blue and sky blue areas are treated independently unless they form adjacent qualifying shapes.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    to_change: Set[Tuple[int, int]] = set()
    
    # Step 1: Identify all qualifying shapes
    for color in [1, 8]:  # Blue and sky blue
        identify_qualifying_shapes(input_grid, color, to_change)
    
    # Step 2: Merge adjacent qualifying shapes
    merge_adjacent_shapes(input_grid, to_change)
    
    # Step 3: Convert marked areas to red
    for r, c in to_change:
        new_grid.values[r][c] = 2
    
    return new_grid

def identify_qualifying_shapes(grid: ColoredGrid, color: int, to_change: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == color:
                for height in range(1, 5):
                    for width in range(1, 5):
                        if height * width >= 4 and is_qualifying_shape(grid, r, c, height, width, color):
                            mark_shape(to_change, r, c, height, width)

def is_qualifying_shape(grid: ColoredGrid, r: int, c: int, height: int, width: int, color: int) -> bool:
    rows, cols = grid.get_dimensions()
    if r + height > rows or c + width > cols:
        return False
    return all(grid.values[r+i][c+j] == color for i in range(height) for j in range(width))

def mark_shape(to_change: Set[Tuple[int, int]], r: int, c: int, height: int, width: int):
    for i in range(height):
        for j in range(width):
            to_change.add((r+i, c+j))

def merge_adjacent_shapes(grid: ColoredGrid, to_change: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] in [1, 8]:
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in to_change:
                        to_change.add((r, c))
                        break
