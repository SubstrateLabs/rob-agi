from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_009d5c81(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing the color of the larger shape (color 8)
    based on its characteristics, removing the smaller shape (color 1),
    and keeping the rest of the grid black (color 0).

    The new color of the larger shape is determined as follows:
    - Red (2) if the shape has more straight lines and angles
    - Green (3) if the shape has more curves and organic forms
    - Orange (7) if the shape is particularly complex and serpentine

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed output grid
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Find all cells with color 8 (larger shape)
    color_8_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 8]

    # Analyze the shape characteristics
    lines = input_grid.detect_lines()
    straight_lines = sum(1 for line in lines if len(line[1]) > 2)
    connected_regions = input_grid.find_connected_regions(8)
    complexity = len(connected_regions)

    # Determine the new color based on shape characteristics
    if complexity > 3 or len(color_8_cells) / (rows * cols) > 0.3:
        new_color = 7  # Orange for complex or large shapes
    elif straight_lines > len(lines) // 2:
        new_color = 2  # Red for more straight lines
    else:
        new_color = 3  # Green for more curves and organic forms

    # Transform the grid
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                output_grid.values[r][c] = new_color
            # Color 1 cells and color 0 cells remain 0 (black) in the output grid

    return output_grid
