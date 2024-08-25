from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and rearranging internal shapes.
    
    1. Extracts the border and internal shapes.
    2. Creates a new grid with 6 columns and the same number of rows as the input.
    3. Places the border in the output grid.
    4. Analyzes the border shape to determine arrangement direction.
    5. Sorts and distributes internal shapes within the border.
    6. Fills any remaining space with the border color.
    
    Returns the transformed ColoredGrid.
    """
    # Extract border and internal shapes
    border_color = input_grid.values[0][0]
    rows, cols = input_grid.get_dimensions()
    border = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == border_color]
    internal_shapes = []
    for color in range(10):
        if color != border_color:
            shapes = input_grid.find_connected_regions(color)
            internal_shapes.extend([(shape, color) for shape in shapes])

    # Create output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(6)] for _ in range(rows)])

    # Place border in output grid
    for r, c in border:
        if c < 6:
            output_grid.values[r][c] = border_color

    # Analyze border shape
    top_empty = next(r for r in range(rows) if output_grid.values[r][1] == 0)
    bottom_empty = next(r for r in range(rows-1, -1, -1) if output_grid.values[r][1] == 0)
    arrange_top_to_bottom = top_empty < rows - 1 - bottom_empty

    # Sort internal shapes
    internal_shapes.sort(key=lambda x: min(c for r, c in x[0]), reverse=not arrange_top_to_bottom)

    # Calculate available space and spacing
    available_space = bottom_empty - top_empty - 1
    spacing = max(1, available_space // (len(internal_shapes) + 1))
    current_row = top_empty + spacing if arrange_top_to_bottom else bottom_empty - spacing

    # Place shapes
    for shape, color in internal_shapes:
        height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
        width = max(c for _, c in shape) - min(c for _, c in shape) + 1
        align_left = len(internal_shapes) % 2 == 0
        
        if align_left:
            col = next(c for c in range(1, 5) if all(output_grid.values[current_row][c:c+width] == [0]*width))
        else:
            col = next(c for c in range(4, 0, -1) if all(output_grid.values[current_row][c-width+1:c+1] == [0]*width))
            col = col - width + 1

        for r, c in shape:
            rr = current_row + (r - min(r for r, _ in shape))
            cc = col + (c - min(c for _, c in shape))
            if 0 <= cc < 5:
                output_grid.values[rr][cc] = color

        if arrange_top_to_bottom:
            current_row += height + spacing
        else:
            current_row -= height + spacing

    # Fill remaining space
    for r in range(rows):
        for c in range(6):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color

    return output_grid
