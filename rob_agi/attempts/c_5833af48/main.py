from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5833af48(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a symmetrical pattern based on the following steps:
    1. Remove the black border and identify the non-border area.
    2. Determine the background color (dominant color in the large rectangle).
    3. Analyze patterns in the top-left and top-right corners.
    4. Calculate the output grid size based on the complexity of the input pattern.
    5. Create an initial output grid filled with the background color.
    6. Transform the input patterns into a symmetrical sky blue (8) pattern.
    7. Apply the transformed pattern to the output grid, ensuring symmetry.
    8. Refine and balance the pattern, adjusting edges for visual appeal.

    The output grid will contain only two colors: the background color and sky blue (8),
    arranged in a symmetrical pattern that represents a transformed version of the input.
    """
    # Remove border and get non-border dimensions
    non_border = remove_border(input_grid.values)
    rows, cols = len(non_border), len(non_border[0])

    # Determine background color
    background_color = get_background_color(non_border)

    # Calculate output dimensions
    pattern_complexity = sum(row.count(8) for row in non_border[:4])
    out_rows = min(max(6, pattern_complexity), 9)
    out_cols = min(max(9, pattern_complexity + 3), 12)

    # Ensure odd dimensions for perfect symmetry
    out_rows = out_rows if out_rows % 2 == 1 else out_rows + 1
    out_cols = out_cols if out_cols % 2 == 1 else out_cols + 1

    # Create initial output grid
    output = [[background_color for _ in range(out_cols)] for _ in range(out_rows)]

    # Transform and apply pattern
    apply_transformed_pattern(output, non_border)

    # Refine and balance the pattern
    refine_pattern(output)

    return ColoredGrid(values=output)

def get_background_color(grid: List[List[int]]) -> int:
    # Assume the background color is the dominant color in the bottom half of the grid
    bottom_half = grid[len(grid)//2:]
    return max(set(cell for row in bottom_half for cell in row if cell != 0), key=lambda x: sum(row.count(x) for row in bottom_half))

def transform_pattern(grid: List[List[int]]) -> List[Tuple[int, int]]:
    pattern = []
    rows, cols = len(grid), len(grid[0])
    for r in range(min(4, rows)):
        for c in range(cols):
            if grid[r][c] == 8:
                pattern.append((r, c))
    return pattern

def apply_pattern(output: List[List[int]], pattern: List[Tuple[int, int]]):
    rows, cols = len(output), len(output[0])
    center = cols // 2
    for r, c in pattern:
        if r < rows and abs(c - center) < center:
            output[r][center + (c - center)] = 8
            output[r][center - (c - center)] = 8
            output[-r-1][center + (c - center)] = 8
            output[-r-1][center - (c - center)] = 8

def refine_pattern(output: List[List[int]]):
    rows, cols = len(output), len(output[0])
    # Ensure first and last rows have sky blue cells
    if 8 not in output[0]:
        output[0][cols//2] = 8
    if 8 not in output[-1]:
        output[-1][cols//2] = 8
    # Ensure first and last columns have sky blue cells
    if 8 not in [output[r][0] for r in range(rows)]:
        output[rows//2][0] = 8
    if 8 not in [output[r][-1] for r in range(rows)]:
        output[rows//2][-1] = 8

def remove_border(grid: List[List[int]]) -> List[List[int]]:
    return [row[1:-1] for row in grid[1:-1] if any(cell != 0 for cell in row)]

def get_dominant_color(grid: List[List[int]]) -> int:
    flat = [cell for row in grid for cell in row if cell != 0]
    return max(set(flat), key=flat.count)

def apply_transformed_pattern(output: List[List[int]], input_pattern: List[List[int]]):
    rows, cols = len(output), len(output[0])
    center_row, center_col = rows // 2, cols // 2

    # Create a basic symmetrical pattern
    for r in range(min(4, len(input_pattern))):
        for c in range(min(5, len(input_pattern[0]))):
            if input_pattern[r][c] == 8:
                # Apply symmetrically in all four quadrants
                output[center_row - r][center_col - c] = 8
                output[center_row - r][center_col + c] = 8
                output[center_row + r][center_col - c] = 8
                output[center_row + r][center_col + c] = 8

    # Ensure pattern touches all edges
    output[0][center_col] = 8
    output[-1][center_col] = 8
    output[center_row][0] = 8
    output[center_row][-1] = 8

def refine_pattern(output: List[List[int]]):
    rows, cols = len(output), len(output[0])
    center_row, center_col = rows // 2, cols // 2

    # Add diagonal elements
    output[1][1] = output[-2][1] = output[1][-2] = output[-2][-2] = 8

    # Ensure symmetry and balance
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 8:
                output[rows - r - 1][c] = 8
                output[r][cols - c - 1] = 8
                output[rows - r - 1][cols - c - 1] = 8

    # Additional refinements for aesthetic appeal
    if rows >= 7 and cols >= 9:
        output[center_row - 1][center_col] = output[center_row + 1][center_col] = 8
        output[center_row][center_col - 1] = output[center_row][center_col + 1] = 8
