from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_b4a43f3b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an 18x18 output grid based on the following steps:
    1. Extracts the top 6x6 pattern and analyzes it.
    2. Identifies full-row colors and the bottom shape.
    3. Determines the scaling factor and arrangement pattern.
    4. Creates a base output grid with scaled and repeated patterns.
    5. Processes full-row colors and the bottom shape.
    6. Fine-tunes the output for balance and coherence.
    7. Ensures a visually appealing expansion of the input pattern.

    The function adapts to various input patterns, prioritizing visual coherence
    and balance in the output while maintaining the essence of the input pattern.
    """
    upper_part = input_grid.values[:6]
    full_row_color = identify_full_row_color(input_grid.values[6])
    lower_part = input_grid.values[8:]

    template = create_template(upper_part)
    scaling_factor = determine_scaling_factor(template)
    arrangement = determine_arrangement(scaling_factor)
    output_grid = create_base_output_grid(template, arrangement, scaling_factor)
    
    if full_row_color:
        apply_full_row_color(output_grid, full_row_color, arrangement)
    
    process_bottom_shape(output_grid, lower_part, scaling_factor)
    center_pattern(output_grid)

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

def identify_full_row_color(row: List[int]) -> int:
    return row[0] if len(set(row)) == 1 and row[0] != 0 else 0

def determine_scaling_factor(template: List[List[int]]) -> int:
    non_zero_count = sum(1 for row in template for cell in row if cell != 0)
    if non_zero_count <= 4:
        return 3
    elif non_zero_count <= 6:
        return 2
    else:
        return 1

def determine_arrangement(scaling_factor: int) -> List[Tuple[int, int]]:
    if scaling_factor == 3:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    elif scaling_factor == 2:
        return [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    else:
        return [(r, c) for r in range(4) for c in range(4)]

def create_base_output_grid(template: List[List[int]], arrangement: List[Tuple[int, int]], scaling_factor: int) -> List[List[int]]:
    output_grid = [[0 for _ in range(18)] for _ in range(18)]
    for ar, ac in arrangement:
        for i in range(3):
            for j in range(3):
                value = template[i][j]
                for si in range(scaling_factor):
                    for sj in range(scaling_factor):
                        r = 3 + ar * 3 * scaling_factor + i * scaling_factor + si
                        c = 3 + ac * 3 * scaling_factor + j * scaling_factor + sj
                        if 0 <= r < 18 and 0 <= c < 18:
                            output_grid[r][c] = value
    return output_grid

def apply_full_row_color(output_grid: List[List[int]], color: int, arrangement: List[Tuple[int, int]]) -> None:
    max_row = max(ar for ar, _ in arrangement) * 3 + 3
    for r in range(3, 15):
        if r % 3 == 2 and r < max_row:
            for c in range(3, 15):
                output_grid[r][c] = color

def process_bottom_shape(output_grid: List[List[int]], lower_part: List[List[int]], scaling_factor: int) -> None:
    shape = [(r, c) for r, row in enumerate(lower_part) for c, val in enumerate(row) if val != 0]
    if not shape:
        return
    
    shape_height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
    shape_width = max(c for _, c in shape) - min(c for _, c in shape) + 1
    
    if shape_height * scaling_factor <= 3 and shape_width * scaling_factor <= 12:
        start_row = 15 - shape_height * scaling_factor
        start_col = 9 - (shape_width * scaling_factor) // 2
        for r, c in shape:
            color = lower_part[r][c]
            for sr in range(scaling_factor):
                for sc in range(scaling_factor):
                    output_row = start_row + r * scaling_factor + sr
                    output_col = start_col + c * scaling_factor + sc
                    if 0 <= output_row < 18 and 0 <= output_col < 18:
                        output_grid[output_row][output_col] = color

def center_pattern(output_grid: List[List[int]]) -> None:
    rows_to_shift = (18 - max((r for r, row in enumerate(output_grid) if any(cell != 0 for cell in row)), default=0)) // 2
    cols_to_shift = (18 - max((c for row in output_grid for c, cell in enumerate(row) if cell != 0), default=0)) // 2
    
    if rows_to_shift > 0 or cols_to_shift > 0:
        new_grid = [[0 for _ in range(18)] for _ in range(18)]
        for r, row in enumerate(output_grid):
            for c, value in enumerate(row):
                if value != 0:
                    new_r = r + rows_to_shift
                    new_c = c + cols_to_shift
                    if 0 <= new_r < 18 and 0 <= new_c < 18:
                        new_grid[new_r][new_c] = value
        output_grid[:] = new_grid
