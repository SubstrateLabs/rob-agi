from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9def23fe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding the red (2) rectangle and creating a comb-like pattern.
    
    1. Identifies the original red rectangle and scattered colored dots.
    2. Creates vertical bars of red starting from the left edge of the original rectangle, including adjacent columns.
    3. Expands the red area horizontally across the width determined by the vertical bars.
    4. Fills the space between adjacent or nearly adjacent vertical bars.
    5. Preserves the positions of all scattered colored dots from the original grid.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Analyze input grid
    original_rect = find_original_rectangle(input_grid)
    scattered_dots = find_scattered_dots(input_grid)

    # Step 2: Create new grid
    new_grid = create_empty_grid(input_grid.get_dimensions())

    # Step 3: Determine vertical bar positions
    vertical_columns = calculate_vertical_columns(original_rect)

    # Step 4: Create vertical bars
    create_vertical_bars(new_grid, vertical_columns)

    # Step 5: Determine and perform horizontal expansion
    expand_horizontally(new_grid, original_rect, vertical_columns)

    # Step 6: Fill between vertical bars
    fill_between_bars(new_grid, vertical_columns, original_rect)

    # Step 7: Preserve scattered dots
    preserve_scattered_dots(new_grid, scattered_dots)

    # Step 8: Return new grid
    return ColoredGrid(values=new_grid)

def find_original_rectangle(grid: ColoredGrid) -> Tuple[int, int, int, int]:
    rows, cols = grid.get_dimensions()
    top = left = bottom = right = -1
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2:
                if top == -1:
                    top = r
                    left = c
                bottom = max(bottom, r)
                right = max(right, c)
    return (top, left, bottom, right)

def find_scattered_dots(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
    rows, cols = grid.get_dimensions()
    dots = []
    for r in range(rows):
        for c in range(cols):
            value = grid.get_cell(r, c)
            if value not in [0, 2]:
                dots.append((r, c, value))
    return dots

def create_empty_grid(dimensions: Tuple[int, int]) -> List[List[int]]:
    rows, cols = dimensions
    return [[0 for _ in range(cols)] for _ in range(rows)]

def expand_horizontally(grid: List[List[int]], rect: Tuple[int, int, int, int], dots: List[Tuple[int, int, int]]):
    top, _, bottom, _ = rect
    rows, cols = len(grid), len(grid[0])
    for r in range(top, bottom + 1):
        for c in range(cols):
            grid[r][c] = 2
    for r, c, value in dots:
        if top <= r <= bottom:
            grid[r][c] = value

def calculate_vertical_columns(rect: Tuple[int, int, int, int]) -> List[int]:
    _, left, _, right = rect
    columns = [left]
    for c in range(left + 1, right + 1):
        if c == right or c - columns[-1] >= 2:
            columns.append(c)
    return columns

def create_vertical_bars(grid: List[List[int]], columns: List[int]):
    rows = len(grid)
    for c in columns:
        for r in range(rows):
            grid[r][c] = 2

def expand_horizontally(grid: List[List[int]], rect: Tuple[int, int, int, int], columns: List[int]):
    top, _, bottom, _ = rect
    left_expand = columns[0]
    right_expand = columns[-1]
    if right_expand == rect[3]:  # If the rightmost bar is at the right edge of the original rectangle
        right_expand = min(right_expand + 1, len(grid[0]) - 1)
    for r in range(top, bottom + 1):
        for c in range(left_expand, right_expand + 1):
            grid[r][c] = 2

def fill_between_bars(grid: List[List[int]], columns: List[int], rect: Tuple[int, int, int, int]):
    rows = len(grid)
    for i in range(len(columns) - 1):
        if columns[i+1] - columns[i] <= 2:
            for r in range(rows):
                for c in range(columns[i], columns[i+1] + 1):
                    grid[r][c] = 2

def preserve_scattered_dots(grid: List[List[int]], dots: List[Tuple[int, int, int]]):
    for r, c, value in dots:
        grid[r][c] = value
