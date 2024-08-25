from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_b4a43f3b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into an 18x18 output grid based on the following steps:
    1. Analyzes the input grid structure, including top pattern, full-width lines, and bottom shape.
    2. Creates a template based on the top 6x6 section.
    3. Determines the appropriate scaling and arrangement based on the input pattern.
    4. Constructs the output grid by repeating and scaling the template.
    5. Incorporates full-width lines and bottom shapes into the output.
    6. Adjusts the final pattern for balance and visual coherence.

    The function adapts to various input patterns, prioritizing visual coherence
    and balance in the output while maintaining the essence of the input pattern.
    """
    upper_part = input_grid.values[:6]
    full_row_colors = identify_full_row_colors(input_grid.values[6:8])
    lower_part = input_grid.values[8:]

    template = create_template(upper_part)
    scaling_factor = determine_scaling_factor(template)
    arrangement = determine_arrangement(template, scaling_factor)
    output_grid = create_base_output_grid(template, arrangement, scaling_factor)
    
    apply_full_row_colors(output_grid, full_row_colors, arrangement)
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

def identify_full_row_colors(rows: List[List[int]]) -> List[int]:
    return [row[0] if len(set(row)) == 1 and row[0] != 0 else 0 for row in rows]

def determine_scaling_factor(template: List[List[int]]) -> int:
    non_zero_count = sum(1 for row in template for cell in row if cell != 0)
    if non_zero_count <= 4:
        return 3
    elif non_zero_count <= 6:
        return 2
    else:
        return 1

def determine_arrangement(template: List[List[int]], scaling_factor: int) -> List[Tuple[int, int]]:
    non_zero_cells = [(r, c) for r in range(3) for c in range(3) if template[r][c] != 0]
    if len(non_zero_cells) <= 4:
        return [(0, 0), (0, 2), (2, 0), (2, 2)]
    elif len(non_zero_cells) <= 6:
        return [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)]
    else:
        return [(r, c) for r in range(3) for c in range(3)]

def create_base_output_grid(template: List[List[int]], arrangement: List[Tuple[int, int]], scaling_factor: int) -> List[List[int]]:
    output_grid = [[0 for _ in range(18)] for _ in range(18)]
    for ar, ac in arrangement:
        for i in range(3):
            for j in range(3):
                value = template[i][j]
                for si in range(scaling_factor):
                    for sj in range(scaling_factor):
                        r = ar * 3 * scaling_factor + i * scaling_factor + si
                        c = ac * 3 * scaling_factor + j * scaling_factor + sj
                        if 0 <= r < 18 and 0 <= c < 18:
                            output_grid[r][c] = value
    return output_grid

def apply_full_row_colors(output_grid: List[List[int]], colors: List[int], arrangement: List[Tuple[int, int]]) -> None:
    max_row = max(ar for ar, _ in arrangement) * 3 + 3
    for idx, color in enumerate(colors):
        if color != 0:
            row = (max_row + idx) % 18
            for c in range(18):
                output_grid[row][c] = color

def process_bottom_shape(output_grid: List[List[int]], lower_part: List[List[int]], scaling_factor: int) -> None:
    shape = [(r, c) for r, row in enumerate(lower_part) for c, val in enumerate(row) if val != 0]
    if not shape:
        return
    
    shape_height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
    shape_width = max(c for _, c in shape) - min(c for _, c in shape) + 1
    
    start_row = 18 - shape_height * scaling_factor
    start_col = (18 - shape_width * scaling_factor) // 2
    
    for r, c in shape:
        color = lower_part[r][c]
        for sr in range(scaling_factor):
            for sc in range(scaling_factor):
                output_row = start_row + r * scaling_factor + sr
                output_col = start_col + c * scaling_factor + sc
                if 0 <= output_row < 18 and 0 <= output_col < 18:
                    output_grid[output_row][output_col] = color

def center_pattern(output_grid: List[List[int]]) -> None:
    rows = [r for r, row in enumerate(output_grid) if any(cell != 0 for cell in row)]
    cols = [c for c in range(18) if any(row[c] != 0 for row in output_grid)]
    
    if not rows or not cols:
        return
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    v_shift = (18 - height) // 2 - min_row
    h_shift = (18 - width) // 2 - min_col
    
    if v_shift != 0 or h_shift != 0:
        new_grid = [[0 for _ in range(18)] for _ in range(18)]
        for r in range(18):
            for c in range(18):
                if output_grid[r][c] != 0:
                    new_r = r + v_shift
                    new_c = c + h_shift
                    if 0 <= new_r < 18 and 0 <= new_c < 18:
                        new_grid[new_r][new_c] = output_grid[r][c]
        output_grid[:] = new_grid
