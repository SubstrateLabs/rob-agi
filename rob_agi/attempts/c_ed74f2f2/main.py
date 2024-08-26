from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Divides the input grid into nine 3x3 sections (with some overlap for the bottom row).
    2. Analyzes each section for the presence of gray (5) cells.
    3. Creates a 3x3 boolean grid marking sections with sufficient gray cells.
    4. Counts the number of True values in the boolean grid.
    5. Determines the final color based on the count:
       - If count is 1, 2, or 9, use green (3)
       - If count is 3 or 4, use red (2)
       - If count is between 5 and 8 (inclusive), use blue (1)
    6. Creates an initial 3x3 ColoredGrid output with the determined color.
    7. Removes "internal" cells (cells surrounded by the same color on all sides) by setting them to black (0).
    8. Returns the final 3x3 ColoredGrid output.
    """
    boolean_grid = analyze_grid(input_grid)
    count = count_true_cells(boolean_grid)
    color = determine_color(count)
    initial_output = create_initial_output(boolean_grid, color)
    return remove_internal_cells(initial_output)

def analyze_grid(grid: ColoredGrid) -> List[List[bool]]:
    boolean_grid = []
    for i in range(3):
        row = []
        for j in range(3):
            section = grid.extract_subgrid(i*2, j*3, 3, 3)
            row.append(analyze_section(section))
        boolean_grid.append(row)
    return boolean_grid

def analyze_section(section: ColoredGrid) -> bool:
    return sum(cell == 5 for row in section.values for cell in row) >= 2

def count_true_cells(boolean_grid: List[List[bool]]) -> int:
    return sum(sum(row) for row in boolean_grid)

def determine_color(count: int) -> int:
    if count in [1, 2, 9]:
        return 3  # Green
    elif count in [3, 4]:
        return 2  # Red
    else:
        return 1  # Blue

def create_initial_output(boolean_grid: List[List[bool]], color: int) -> ColoredGrid:
    return ColoredGrid(values=[[color if cell else 0 for cell in row] for row in boolean_grid])

def remove_internal_cells(grid: ColoredGrid) -> ColoredGrid:
    new_values = [row[:] for row in grid.values]
    for r in range(3):
        for c in range(3):
            if is_internal_cell(grid, r, c):
                new_values[r][c] = 0
    return ColoredGrid(values=new_values)

def is_internal_cell(grid: ColoredGrid, r: int, c: int) -> bool:
    if grid.values[r][c] == 0:
        return False
    color = grid.values[r][c]
    neighbors = get_neighbors(r, c)
    return all(0 <= nr < 3 and 0 <= nc < 3 and grid.values[nr][nc] == color for nr, nc in neighbors)

def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
