from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9def23fe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding the red (2) rectangle and creating a comb-like pattern.
    
    1. Identifies the original red rectangle and scattered colored dots.
    2. Determines vertical bar positions based on the original rectangle's width.
    3. Determines horizontal bar positions based on the original rectangle's height.
    4. Creates a new grid with expanded red area, including horizontal and vertical bars.
    5. Fills the area between the leftmost and rightmost vertical bars from the top of the original rectangle to the bottom of the grid.
    6. Preserves the positions of all scattered colored dots from the original grid.
    
    The transformation includes:
    - Creating vertical red bars at the left and right edges of the original rectangle, and 1-2 additional bars based on the width.
    - Creating horizontal red bars at the top and bottom of the original rectangle, and potentially a middle bar for taller rectangles.
    - Extending the red area vertically from the top of the original rectangle to the bottom of the grid, between the outermost vertical bars.
    - Maintaining all non-red, non-black dots from the original grid in their original positions.
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    # Step 1: Analyze input grid
    original_rect = find_original_rectangle(input_grid)
    scattered_dots = find_scattered_dots(input_grid)

    # Step 2: Create new grid
    new_grid = create_empty_grid(input_grid.get_dimensions())

    # Step 3: Determine vertical and horizontal bar positions
    vertical_bars = calculate_vertical_bars(original_rect)
    horizontal_bars = calculate_horizontal_bars(original_rect)

    # Step 4: Draw vertical and horizontal bars
    draw_vertical_bars(new_grid, vertical_bars)
    draw_horizontal_bars(new_grid, horizontal_bars)

    # Step 5: Fill expanded rectangle
    fill_expanded_rectangle(new_grid, original_rect, vertical_bars)

    # Step 6: Preserve scattered dots
    preserve_scattered_dots(new_grid, scattered_dots)

    # Step 7: Return new grid
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

def calculate_vertical_bars(rect: Tuple[int, int, int, int]) -> List[int]:
    _, left, _, right = rect
    width = right - left + 1
    bars = [left, right]
    if width <= 6:
        bars.append(left + width // 2)
    else:
        bars.extend([left + width // 3, left + 2 * width // 3])
    return sorted(bars)

def calculate_horizontal_bars(rect: Tuple[int, int, int, int]) -> List[int]:
    top, _, bottom, _ = rect
    height = bottom - top + 1
    bars = [top, bottom]
    if height > 5:
        bars.append((top + bottom) // 2)
    return sorted(bars)

def draw_vertical_bars(grid: List[List[int]], bars: List[int]):
    for c in bars:
        for r in range(len(grid)):
            grid[r][c] = 2

def draw_horizontal_bars(grid: List[List[int]], bars: List[int]):
    for r in bars:
        grid[r] = [2] * len(grid[0])

def fill_expanded_rectangle(grid: List[List[int]], rect: Tuple[int, int, int, int], vertical_bars: List[int]):
    top, _, _, _ = rect
    rows, cols = len(grid), len(grid[0])
    left, right = min(vertical_bars), max(vertical_bars)
    for r in range(top, rows):
        for c in range(left, right + 1):
            grid[r][c] = 2

def preserve_scattered_dots(grid: List[List[int]], dots: List[Tuple[int, int, int]]):
    for r, c, value in dots:
        grid[r][c] = value
