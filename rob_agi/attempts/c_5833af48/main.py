from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5833af48(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a symmetrical pattern based on the following steps:
    1. Analyze the input grid to identify the background color and extract the L-shape pattern.
    2. Calculate the output grid size based on the input pattern and grid dimensions.
    3. Create an initial output grid filled with the background color.
    4. Place L-shape patterns in the corners, ensuring they touch the edges.
    5. Add edge connections and central elements to create a symmetrical pattern.
    6. Refine the pattern to ensure perfect symmetry and balance.

    The output grid will contain only two colors: the background color and sky blue (8),
    arranged in a symmetrical pattern that represents a simplified, transformed version of the input.
    """
    background_color = get_background_color(input_grid.values)
    pattern = extract_l_shape(input_grid.values)
    out_rows, out_cols = calculate_output_size(input_grid.values, pattern)
    output = [[background_color for _ in range(out_cols)] for _ in range(out_rows)]
    apply_corner_patterns(output, pattern)
    add_edge_connections(output)
    add_central_elements(output)
    refine_pattern(output)
    return ColoredGrid(values=output)

def get_background_color(grid: List[List[int]]) -> int:
    colors = [cell for row in grid for cell in row if cell != 0 and cell != 8]
    return max(set(colors), key=colors.count)

def extract_l_shape(grid: List[List[int]]) -> List[Tuple[int, int]]:
    pattern = []
    for r in range(min(5, len(grid))):
        for c in range(min(5, len(grid[0]))):
            if grid[r][c] == 8:
                pattern.append((r, c))
    return pattern

def calculate_output_size(grid: List[List[int]], pattern: List[Tuple[int, int]]) -> Tuple[int, int]:
    input_rows, input_cols = len(grid), len(grid[0])
    pattern_size = max(max(r for r, _ in pattern), max(c for _, c in pattern)) + 1
    base_size = pattern_size * 2 + 1
    rows = max(base_size, 9)
    cols = max(base_size, 9)
    if input_cols > input_rows:
        cols += 2
    elif input_rows > input_cols:
        rows += 2
    return rows + (rows % 2 == 0), cols + (cols % 2 == 0)

def apply_corner_patterns(output: List[List[int]], pattern: List[Tuple[int, int]]):
    rows, cols = len(output), len(output[0])
    for r, c in pattern:
        output[r][c] = output[r][cols-1-c] = output[rows-1-r][c] = output[rows-1-r][cols-1-c] = 8

def add_edge_connections(output: List[List[int]]):
    rows, cols = len(output), len(output[0])
    mid_row, mid_col = rows // 2, cols // 2
    output[0][mid_col] = output[rows-1][mid_col] = output[mid_row][0] = output[mid_row][cols-1] = 8

def add_central_elements(output: List[List[int]]):
    rows, cols = len(output), len(output[0])
    mid_row, mid_col = rows // 2, cols // 2
    output[mid_row][mid_col] = 8
    if rows >= 9 and cols >= 9:
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            output[mid_row+dr][mid_col+dc] = 8

def refine_pattern(output: List[List[int]]):
    rows, cols = len(output), len(output[0])
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 8:
                output[rows-1-r][c] = output[r][cols-1-c] = output[rows-1-r][cols-1-c] = 8

    # Remove isolated cells
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            neighbors = sum(output[r+dr][c+dc] == 8 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)])
            if output[r][c] == 8 and neighbors < 2:
                output[r][c] = output[0][0]  # Change to background color

    # Ensure perfect symmetry
    for r in range(rows // 2 + 1):
        for c in range(cols):
            if output[r][c] == 8:
                output[rows-1-r][c] = output[r][cols-1-c] = output[rows-1-r][cols-1-c] = 8
