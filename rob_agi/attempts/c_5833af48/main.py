from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5833af48(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a symmetrical pattern based on the following steps:
    1. Remove the black border and identify the non-border area.
    2. Determine the background color (dominant color in the large rectangle).
    3. Analyze the pattern in the top-left corner.
    4. Calculate the output grid size based on the complexity of the input pattern.
    5. Create an initial output grid filled with the background color.
    6. Generate a symmetrical sky blue (8) pattern based on the input pattern.
    7. Apply the pattern to the output grid, ensuring symmetry and edge/corner touching.
    8. Refine and balance the pattern for visual appeal and perfect symmetry.

    The output grid will contain only two colors: the background color and sky blue (8),
    arranged in a symmetrical pattern that represents a simplified, transformed version of the input.
    """
    non_border = remove_border(input_grid.values)
    background_color = get_background_color(non_border)
    pattern = analyze_pattern(non_border)
    out_rows, out_cols = calculate_output_size(pattern)
    output = [[background_color for _ in range(out_cols)] for _ in range(out_rows)]
    apply_symmetrical_pattern(output, pattern)
    refine_pattern(output)
    return ColoredGrid(values=output)

def get_background_color(grid: List[List[int]]) -> int:
    return max(set(cell for row in grid for cell in row if cell != 0 and cell != 8), 
               key=lambda x: sum(row.count(x) for row in grid))

def remove_border(grid: List[List[int]]) -> List[List[int]]:
    return [row[1:-1] for row in grid[1:-1] if any(cell != 0 for cell in row)]

def analyze_pattern(grid: List[List[int]]) -> List[Tuple[int, int]]:
    pattern = []
    for r in range(min(5, len(grid))):
        for c in range(min(5, len(grid[0]))):
            if grid[r][c] == 8:
                pattern.append((r, c))
    return pattern

def calculate_output_size(pattern: List[Tuple[int, int]]) -> Tuple[int, int]:
    complexity = len(pattern)
    size = max(6, min(9, complexity + 3))
    return size, size + 3  # Make it slightly wider for better symmetry

def apply_symmetrical_pattern(output: List[List[int]], pattern: List[Tuple[int, int]]):
    rows, cols = len(output), len(output[0])
    center_row, center_col = rows // 2, cols // 2

    # Apply pattern to corners
    for r, c in pattern:
        dr, dc = r - 2, c - 2  # Adjust for centering
        for sr, sc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = center_row + sr * dr, center_col + sc * dc
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr][nc] = 8

    # Ensure pattern touches all edges and corners
    output[0][0] = output[0][-1] = output[-1][0] = output[-1][-1] = 8
    output[0][center_col] = output[-1][center_col] = 8
    output[center_row][0] = output[center_row][-1] = 8

    # Add central feature
    output[center_row][center_col] = 8

def refine_pattern(output: List[List[int]]):
    rows, cols = len(output), len(output[0])

    # Connect edge patterns
    for r in range(1, rows - 1):
        if output[r][0] == 8 and output[r][-1] == 8:
            output[r][cols // 4] = output[r][cols // 2] = output[r][3 * cols // 4] = 8
    for c in range(1, cols - 1):
        if output[0][c] == 8 and output[-1][c] == 8:
            output[rows // 4][c] = output[rows // 2][c] = output[3 * rows // 4][c] = 8

    # Ensure symmetry
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 8:
                output[rows - r - 1][c] = output[r][cols - c - 1] = output[rows - r - 1][cols - c - 1] = 8

    # Remove isolated cells
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if output[r][c] == 8 and sum(output[r+dr][c+dc] == 8 for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]) < 2:
                output[r][c] = output[0][0]  # Change to background color
