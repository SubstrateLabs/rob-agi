from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9def23fe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding the red (2) rectangle and creating a comb-like pattern.
    
    1. Identifies the original red rectangle and scattered colored dots.
    2. Determines horizontal and vertical expansion based on the original rectangle's dimensions.
    3. Creates a new grid with expanded red area, including horizontal bars and vertical columns.
    4. Fills the space between adjacent or nearly adjacent vertical bars.
    5. Extends the red area within the bounds of the expansion.
    6. Preserves the positions of all scattered colored dots from the original grid.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Analyze input grid
    original_rect = find_original_rectangle(input_grid)
    scattered_dots = find_scattered_dots(input_grid)

    # Step 2: Create new grid
    new_grid = create_empty_grid(input_grid.get_dimensions())

    # Step 3: Determine horizontal and vertical expansion
    horizontal_bars = calculate_horizontal_bars(original_rect)
    vertical_columns = calculate_vertical_columns(original_rect)

    # Step 4: Apply horizontal and vertical expansion
    apply_horizontal_expansion(new_grid, horizontal_bars)
    apply_vertical_expansion(new_grid, vertical_columns)

    # Step 5: Fill between vertical bars
    fill_between_bars(new_grid, vertical_columns, original_rect)

    # Step 6: Extend red area
    extend_red_area(new_grid, horizontal_bars, vertical_columns)

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

def calculate_horizontal_bars(rect: Tuple[int, int, int, int]) -> List[int]:
    top, _, bottom, _ = rect
    height = bottom - top + 1
    if height <= 5:
        return [top, bottom]
    else:
        middle = (top + bottom) // 2
        return [top, middle, bottom]

def calculate_vertical_columns(rect: Tuple[int, int, int, int]) -> List[int]:
    _, left, _, right = rect
    width = right - left + 1
    columns = [left + i * (width // (width + 1)) for i in range(width + 2)]
    return [c for c in columns if c < len(rect[0])]

def apply_horizontal_expansion(grid: List[List[int]], bars: List[int]):
    for r in bars:
        grid[r] = [2] * len(grid[0])

def apply_vertical_expansion(grid: List[List[int]], columns: List[int]):
    for c in columns:
        for r in range(len(grid)):
            grid[r][c] = 2

def extend_red_area(grid: List[List[int]], horizontal_bars: List[int], vertical_columns: List[int]):
    top, bottom = min(horizontal_bars), max(horizontal_bars)
    left, right = min(vertical_columns), max(vertical_columns)
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
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
