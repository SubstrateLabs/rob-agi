from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ed74f2f2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 9x5 input grid into a 3x3 output grid based on the following rules:
    1. Divides the input grid into nine 3x3 sections.
    2. Analyzes each section for the presence of gray (5) cells.
    3. Creates an initial 3x3 output grid marking sections with sufficient gray cells.
    4. Analyzes the pattern in the initial output grid.
    5. Determines the final color based on the recognized pattern:
       - 'C' or 'L' shapes (including rotations) use red (2)
       - Line patterns use blue (1)
       - Complex or branching patterns use green (3)
    6. Creates the final 3x3 ColoredGrid output with the determined color.
    """
    sections = divide_grid(input_grid)
    analyzed_sections = [analyze_section(section) for section in sections]
    initial_output = create_initial_output(analyzed_sections)
    color = determine_color(initial_output)
    return create_final_output(initial_output, color)

def divide_grid(grid: ColoredGrid) -> List[ColoredGrid]:
    sections = []
    for i in range(3):
        for j in range(3):
            section = grid.extract_subgrid(i*3, j*3, 3, 3)
            sections.append(section)
    return sections

def analyze_section(section: ColoredGrid) -> bool:
    return sum(cell == 5 for row in section.values for cell in row) >= 2

def create_initial_output(analyzed_sections: List[bool]) -> List[List[int]]:
    return [[int(analyzed_sections[i*3 + j]) for j in range(3)] for i in range(3)]

def is_c_or_l_shape(grid: List[List[int]]) -> bool:
    patterns = [
        [[1,1,1],[1,0,0],[1,1,1]],  # C
        [[1,1,1],[1,0,0],[1,0,0]],  # L
        [[1,1,1],[0,0,1],[1,1,1]],  # Reversed C
        [[0,0,1],[0,0,1],[1,1,1]],  # Rotated L
    ]
    return any(grid == pattern for pattern in patterns)

def is_line_pattern(grid: List[List[int]]) -> bool:
    return (any(sum(row) == 3 for row in grid) or  # Horizontal
            any(sum(col) == 3 for col in zip(*grid)) or  # Vertical
            sum(grid[i][i] for i in range(3)) == 3 or  # Diagonal
            sum(grid[i][2-i] for i in range(3)) == 3)  # Other diagonal

def is_complex_pattern(grid: List[List[int]]) -> bool:
    return not (is_c_or_l_shape(grid) or is_line_pattern(grid))

def determine_color(grid: List[List[int]]) -> int:
    if is_c_or_l_shape(grid):
        return 2  # Red
    elif is_line_pattern(grid):
        return 1  # Blue
    else:
        return 3  # Green

def create_final_output(initial_grid: List[List[int]], color: int) -> ColoredGrid:
    return ColoredGrid(values=[[color if cell else 0 for cell in row] for row in initial_grid])
