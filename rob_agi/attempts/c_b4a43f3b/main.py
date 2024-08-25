from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_b4a43f3b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an 18x18 output grid based on the following steps:
    1. Creates a 3x3 template from the top 6x6 part of the input.
    2. Analyzes the bottom part to determine the arrangement pattern.
    3. Calculates the scaling factor based on the arrangement.
    4. Determines the positions for placing the scaled templates.
    5. Creates the output grid by placing scaled templates according to the arrangement.
    6. Ensures a 3-cell black border around the entire pattern.
    7. Handles special cases like centering the pattern if it doesn't fill the space.

    The function dynamically adapts to various input patterns and arrangements,
    scaling the output to fit within the 18x18 grid while maintaining proportions.
    """
    upper_part = input_grid.values[:6]
    lower_part = input_grid.values[7:]

    template = create_template(upper_part)
    arrangement = analyze_arrangement(lower_part)
    scaling_factor = calculate_scaling_factor(arrangement)
    positions = calculate_positions(arrangement, scaling_factor)
    output_grid = create_output_grid(template, positions, scaling_factor)

    return ColoredGrid(values=output_grid)

def create_template(upper_part: List[List[int]]) -> List[List[int]]:
    template = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            block = [upper_part[2*i][2*j], upper_part[2*i][2*j+1],
                     upper_part[2*i+1][2*j], upper_part[2*i+1][2*j+1]]
            counter = Counter(x for x in block if x != 0)
            template[i][j] = counter.most_common(1)[0][0] if counter else 0
    return template

def analyze_arrangement(lower_part: List[List[int]]) -> List[Tuple[int, int]]:
    return [(r, c) for r, row in enumerate(lower_part) for c, val in enumerate(row) if val != 0]

def calculate_scaling_factor(arrangement: List[Tuple[int, int]]) -> int:
    if not arrangement:
        return 1
    height = max(r for r, _ in arrangement) - min(r for r, _ in arrangement) + 1
    width = max(c for _, c in arrangement) - min(c for _, c in arrangement) + 1
    max_dim = max(height, width)
    return max(1, min(12 // max_dim, 3))

def calculate_positions(arrangement: List[Tuple[int, int]], scaling_factor: int) -> List[Tuple[int, int]]:
    if not arrangement:
        return []
    min_r, min_c = min(r for r, _ in arrangement), min(c for _, c in arrangement)
    normalized = [(r - min_r, c - min_c) for r, c in arrangement]
    max_r, max_c = max(r for r, _ in normalized), max(c for _, c in normalized)
    center_offset_r = (12 - (max_r + 1) * scaling_factor) // 2
    center_offset_c = (12 - (max_c + 1) * scaling_factor) // 2
    return [(3 + center_offset_r + r * scaling_factor, 3 + center_offset_c + c * scaling_factor) for r, c in normalized]

def create_output_grid(template: List[List[int]], positions: List[Tuple[int, int]], scaling_factor: int) -> List[List[int]]:
    output_grid = [[0 for _ in range(18)] for _ in range(18)]
    for start_row, start_col in positions:
        for i in range(3):
            for j in range(3):
                value = template[i][j]
                for si in range(scaling_factor):
                    for sj in range(scaling_factor):
                        r = start_row + i * scaling_factor + si
                        c = start_col + j * scaling_factor + sj
                        if 0 <= r < 18 and 0 <= c < 18:
                            output_grid[r][c] = value
    return output_grid
