from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b4a43f3b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an 18x18 output grid based on the following steps:
    1. Extracts the top 6 rows to create a 3x3 template.
    2. Uses rows 8-13 to determine the arrangement pattern.
    3. Creates a 3x3 template from the top rows.
    4. Creates an arrangement map from the bottom rows.
    5. Calculates the scaling factor based on the arrangement.
    6. Places scaled templates in the output grid according to the arrangement.
    7. Ensures a 3-cell black border around the pattern.

    The function dynamically adapts to various input patterns and arrangements,
    scaling the output to fit within the 18x18 grid while maintaining proportions.
    """
    # Extract relevant parts of the input grid
    upper_part = input_grid.values[:6]
    lower_part = input_grid.values[8:13]

    # Create 3x3 template
    template = create_template(upper_part)

    # Create arrangement map
    arrangement_map = create_arrangement_map(lower_part)

    # Calculate scaling factor
    scaling_factor = calculate_scaling_factor(arrangement_map)

    # Create output grid
    output_grid = create_output_grid(template, arrangement_map, scaling_factor)

    return ColoredGrid(values=output_grid)

def create_template(upper_part: List[List[int]]) -> List[List[int]]:
    template = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            block = [upper_part[2*i][2*j], upper_part[2*i][2*j+1],
                     upper_part[2*i+1][2*j], upper_part[2*i+1][2*j+1]]
            non_zero = [x for x in block if x != 0]
            template[i][j] = non_zero[0] if non_zero else 0
    return template

def create_arrangement_map(lower_part: List[List[int]]) -> List[List[bool]]:
    return [[cell != 0 for cell in row] for row in lower_part]

def calculate_scaling_factor(arrangement_map: List[List[bool]]) -> int:
    height = max(sum(row) for row in arrangement_map)
    width = max(sum(col) for col in zip(*arrangement_map))
    return max(1, min(12 // max(height, width), 3))

def create_output_grid(template: List[List[int]], arrangement_map: List[List[bool]], scaling_factor: int) -> List[List[int]]:
    output_grid = [[0 for _ in range(18)] for _ in range(18)]
    start_row, start_col = 3, 3
    for i, row in enumerate(arrangement_map):
        for j, place in enumerate(row):
            if place:
                for ti in range(3):
                    for tj in range(3):
                        value = template[ti][tj]
                        for si in range(scaling_factor):
                            for sj in range(scaling_factor):
                                r = start_row + i * 3 * scaling_factor + ti * scaling_factor + si
                                c = start_col + j * 3 * scaling_factor + tj * scaling_factor + sj
                                if 0 <= r < 18 and 0 <= c < 18:
                                    output_grid[r][c] = value
    return output_grid
