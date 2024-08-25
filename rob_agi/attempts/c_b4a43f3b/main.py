from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b4a43f3b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an 18x18 output grid based on the following steps:
    1. Parses the input grid, separating upper and lower parts.
    2. Creates a pattern template from the upper part.
    3. Analyzes the lower part to determine arrangement (horizontal or vertical).
    4. Creates an arrangement pattern based on the lower part.
    5. Initializes an 18x18 output grid.
    6. Places pattern instances according to the arrangement.
    7. Adds a black border and fills remaining space.

    The function identifies color regions in the upper part, creates 3x3 blocks
    for each region, and arranges them based on the non-black squares in the lower part.
    """
    # Parse input grid
    upper_part = input_grid.values[:6]
    lower_part = input_grid.values[8:13]

    # Create pattern template
    pattern_template = create_pattern_template(upper_part)

    # Analyze lower part and create arrangement pattern
    arrangement = analyze_lower_part(lower_part)
    arrangement_pattern = create_arrangement_pattern(lower_part)

    # Initialize output grid
    output_grid = [[0 for _ in range(18)] for _ in range(18)]

    # Place pattern instances
    place_pattern_instances(output_grid, pattern_template, arrangement_pattern, arrangement)

    # Add border and fill remaining space
    add_border_and_fill(output_grid)

    return ColoredGrid(values=output_grid)

def create_pattern_template(upper_part: List[List[int]]) -> List[List[int]]:
    template = [[0 for _ in range(9)] for _ in range(9)]
    for i in range(0, 6, 2):
        for j in range(0, 6, 2):
            color = upper_part[i][j]
            if color != 0:
                template[i//2*3+1][j//2*3+1] = color
    return template

def analyze_lower_part(lower_part: List[List[int]]) -> str:
    row_counts = [sum(1 for cell in row if cell != 0) for row in lower_part]
    col_counts = [sum(1 for row in lower_part if row[j] != 0) for j in range(6)]
    return "horizontal" if max(row_counts) > max(col_counts) else "vertical"

def create_arrangement_pattern(lower_part: List[List[int]]) -> List[Tuple[int, int]]:
    return [(i, j) for i, row in enumerate(lower_part) for j, cell in enumerate(row) if cell != 0]

def place_pattern_instances(output_grid: List[List[int]], pattern_template: List[List[int]], 
                            arrangement_pattern: List[Tuple[int, int]], arrangement: str):
    for idx, (i, j) in enumerate(arrangement_pattern):
        if arrangement == "horizontal":
            row, col = (idx // 3) * 6, (idx % 3) * 6
        else:
            row, col = (idx % 3) * 6, (idx // 3) * 6
        for r in range(9):
            for c in range(9):
                if 3 <= row+r < 15 and 3 <= col+c < 15:
                    output_grid[row+r][col+c] = pattern_template[r][c]

def add_border_and_fill(output_grid: List[List[int]]):
    for i in range(18):
        for j in range(18):
            if i < 3 or i >= 15 or j < 3 or j >= 15:
                output_grid[i][j] = 0
