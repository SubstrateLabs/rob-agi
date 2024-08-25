from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9def23fe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding the red (2) rectangle and creating a comb-like pattern.
    
    1. Identifies the original red rectangle and scattered colored dots.
    2. Expands the red area horizontally across the full width of the grid where the original rectangle was.
    3. Creates vertical bars of red starting from the left edge of the original rectangle and at every third column.
    4. Fills the space between adjacent or nearly adjacent vertical bars.
    5. Preserves the positions of all scattered colored dots from the original grid.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Analyze input grid
    original_rect = find_original_rectangle(input_grid)
    scattered_dots = find_scattered_dots(input_grid)

    # Step 2: Create new grid
    new_grid = create_empty_grid(input_grid.get_dimensions())

    # Step 3: Expand horizontally
    expand_horizontally(new_grid, original_rect, scattered_dots)

    # Step 4: Determine vertical expansion columns
    vertical_columns = calculate_vertical_columns(original_rect)

    # Step 5: Expand vertically
    expand_vertically(new_grid, vertical_columns, original_rect, scattered_dots)

    # Step 6: Fill between vertical bars
    fill_between_bars(new_grid, vertical_columns, original_rect, scattered_dots)

    # Step 7: Transfer unchanged areas
    transfer_unchanged_areas(new_grid, input_grid)

    # Step 9: Return new grid
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
    for c in range(left + 3, right + 1, 3):
        columns.append(c)
    return columns

def expand_vertically(grid: List[List[int]], columns: List[int], rect: Tuple[int, int, int, int], dots: List[Tuple[int, int, int]]):
    top, _, _, _ = rect
    rows = len(grid)
    for c in columns:
        for r in range(top, rows):
            grid[r][c] = 2
    for r, c, value in dots:
        if r >= top and c in columns:
            grid[r][c] = value

def fill_between_bars(grid: List[List[int]], columns: List[int], rect: Tuple[int, int, int, int], dots: List[Tuple[int, int, int]]):
    top, _, _, _ = rect
    rows, cols = len(grid), len(grid[0])
    for i in range(len(columns) - 1):
        if columns[i+1] - columns[i] <= 2:
            for r in range(top, rows):
                for c in range(columns[i], columns[i+1] + 1):
                    grid[r][c] = 2
    for r, c, value in dots:
        if r >= top:
            grid[r][c] = value

def transfer_unchanged_areas(new_grid: List[List[int]], original_grid: ColoredGrid):
    rows, cols = original_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if new_grid[r][c] == 0:
                new_grid[r][c] = original_grid.get_cell(r, c)
