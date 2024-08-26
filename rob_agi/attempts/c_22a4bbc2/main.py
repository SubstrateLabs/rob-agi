from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

from typing import List, Tuple, Set, Dict

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
    
    # Step 1: Identify largest qualifying shapes
    for color in [1, 8]:  # Blue and sky blue
        for shape in [(3,3), (3,2), (2,3), (2,2), (4,1), (1,4)]:
            identify_shapes(input_grid, color, shape, to_change)
    
    # Step 2: Handle 3x1 and 1x3 shapes
    for color in [1, 8]:
        for shape in [(3,1), (1,3)]:
            identify_adjacent_shapes(input_grid, color, shape, to_change)
    
    # Step 3: Merge adjacent qualifying shapes
    merge_adjacent_shapes(input_grid, to_change)
    
    # Step 4: Convert marked areas to red
    for r, c in to_change:
        new_grid.values[r][c] = 2
    
    # Step 5: Handle isolated 2x2 squares
    handle_isolated_squares(new_grid)
    
    return new_grid

def identify_shapes(grid: ColoredGrid, color: int, shape: Tuple[int, int], to_change: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    height, width = shape
    for r in range(rows - height + 1):
        for c in range(cols - width + 1):
            if all(grid.values[r+i][c+j] == color for i in range(height) for j in range(width)):
                for i in range(height):
                    for j in range(width):
                        to_change.add((r+i, c+j))

def identify_adjacent_shapes(grid: ColoredGrid, color: int, shape: Tuple[int, int], to_change: Set[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    height, width = shape
    for r in range(rows - height + 1):
        for c in range(cols - width + 1):
            if all(grid.values[r+i][c+j] == color for i in range(height) for j in range(width)):
                if is_adjacent_to_qualifying(grid, r, c, height, width, to_change):
                    for i in range(height):
                        for j in range(width):
                            to_change.add((r+i, c+j))

def is_adjacent_to_qualifying(grid: ColoredGrid, r: int, c: int, height: int, width: int, to_change: Set[Tuple[int, int]]) -> bool:
    rows, cols = grid.get_dimensions()
    for i in range(height):
        for j in range(width):
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + i + dr, c + j + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in to_change:
                    return True
    return False

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

def handle_isolated_squares(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.values[r+i][c+j] in [1, 8] for i in range(2) for j in range(2)):
                if is_isolated(grid, r, c):
                    for i in range(2):
                        for j in range(2):
                            grid.values[r+i][c+j] = 2

def is_isolated(grid: ColoredGrid, r: int, c: int) -> bool:
    rows, cols = grid.get_dimensions()
    for i in range(2):
        for j in range(2):
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + i + dr, c + j + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr < r or nr > r+1 or nc < c or nc > c+1):
                    if grid.values[nr][nc] in [1, 8]:
                        return False
    return True
