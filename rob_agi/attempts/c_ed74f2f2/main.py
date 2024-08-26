from rob_agi.colored_grid import ColoredGrid
from typing import List

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Divides the input grid into nine 3x3 sections (with some overlap for the bottom row).
    2. Analyzes each section for the presence of gray (5) cells.
    3. Creates a 3x3 boolean grid marking sections with sufficient gray cells.
    4. Counts the number of True values in the boolean grid.
    5. Determines the final color based on the count:
       - If count is 3 or 4, use red (2)
       - If count is between 5 and 8 (inclusive), use blue (1)
       - If count is 1, 2, or 9, use green (3)
    6. Creates the final 3x3 ColoredGrid output with the determined color.
    """
    boolean_grid = analyze_grid(input_grid)
    count = count_true_cells(boolean_grid)
    color = determine_color(count)
    return create_final_output(boolean_grid, color)

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
    if count in [3, 4]:
        return 2  # Red
    elif 5 <= count <= 8:
        return 1  # Blue
    else:
        return 3  # Green

def create_final_output(boolean_grid: List[List[bool]], color: int) -> ColoredGrid:
    return ColoredGrid(values=[[color if cell else 0 for cell in row] for row in boolean_grid])
