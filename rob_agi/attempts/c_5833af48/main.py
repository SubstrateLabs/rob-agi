from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5833af48(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a symmetrical pattern based on the following steps:
    1. Remove the black border and identify the non-border area.
    2. Determine the dominant color and identify patterns in the top-left corner.
    3. Calculate the output grid size based on the non-border dimensions.
    4. Create an initial output grid filled with the dominant color.
    5. Transform the input patterns into a symmetrical sky blue (8) pattern.
    6. Apply the transformed pattern to the output grid.
    7. Refine and balance the pattern.

    The output grid will contain only two colors: the dominant color and sky blue (8),
    arranged in a symmetrical pattern that represents a transformed version of the input.
    """
    # Remove border and get non-border dimensions
    non_border = remove_border(input_grid.values)
    rows, cols = len(non_border), len(non_border[0])

    # Determine dominant color
    dominant_color = get_dominant_color(non_border)

    # Calculate output dimensions
    out_rows, out_cols = rows // 2, cols - 2

    # Create initial output grid
    output = [[dominant_color for _ in range(out_cols)] for _ in range(out_rows)]

    # Transform and apply pattern
    pattern = transform_pattern(non_border)
    apply_pattern(output, pattern)

    return ColoredGrid(values=output)

def remove_border(grid: List[List[int]]) -> List[List[int]]:
    return [row[1:-1] for row in grid[1:-1] if any(cell != 0 for cell in row)]

def get_dominant_color(grid: List[List[int]]) -> int:
    flat = [cell for row in grid for cell in row if cell != 0]
    return max(set(flat), key=flat.count)

def transform_pattern(grid: List[List[int]]) -> List[Tuple[int, int]]:
    pattern = []
    rows, cols = len(grid), len(grid[0])
    for r in range(min(4, rows)):
        for c in range(min(5, cols)):
            if grid[r][c] == 8:
                pattern.append((r, c))
    return pattern

def apply_pattern(output: List[List[int]], pattern: List[Tuple[int, int]]):
    rows, cols = len(output), len(output[0])
    for r, c in pattern:
        if r < rows and c < cols:
            output[r][c] = 8
        if r < rows and cols - c - 1 >= 0:
            output[r][cols - c - 1] = 8
        if rows - r - 1 >= 0 and c < cols:
            output[rows - r - 1][c] = 8
        if rows - r - 1 >= 0 and cols - c - 1 >= 0:
            output[rows - r - 1][cols - c - 1] = 8
