from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and rearranging internal shapes.
    
    1. Extracts the border from the left 6 columns of the input grid.
    2. Identifies and extracts shapes from the right side of the input grid.
    3. Inverts the extracted shapes vertically.
    4. Places inverted shapes in empty spaces within the border, maintaining their relative horizontal position.
    5. Fills any remaining empty space with the border color.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    border_color = input_grid.values[0][0]
    
    # Extract border
    output_grid = ColoredGrid(values=[[input_grid.values[r][c] for c in range(6)] for r in range(rows)])
    
    # Extract internal shapes
    internal_shapes = []
    for color in range(10):
        if color != border_color and color != 0:
            shapes = input_grid.find_connected_regions(color)
            for shape in shapes:
                if any(c >= 6 for _, c in shape):  # Only consider shapes from the right side
                    internal_shapes.append((shape, color))
    
    # Process shapes: invert vertically and calculate dimensions
    processed_shapes = []
    for shape, color in internal_shapes:
        min_r, max_r = min(r for r, _ in shape), max(r for r, _ in shape)
        min_c, max_c = min(c for _, c in shape), max(c for _, c in shape)
        height, width = max_r - min_r + 1, max_c - min_c + 1
        right_distance = cols - max_c - 1
        inverted_shape = [((rows - 1 - r + min_r, c - min_c), color) for r, c in shape]
        processed_shapes.append((inverted_shape, height, width, right_distance))
    
    # Sort shapes by their original vertical position (top to bottom)
    processed_shapes.sort(key=lambda x: min(r for (r, _), _ in x[0]))
    
    # Place shapes in empty spaces
    for shape, height, width, right_distance in processed_shapes:
        placed = False
        for r in range(rows - height + 1):
            if placed:
                break
            c = 5 - right_distance - width + 1
            if c < 1:
                c = 1
            if all(output_grid.values[r+dr][c+dc] == 0 for (dr, dc), _ in shape):
                for (dr, dc), color in shape:
                    output_grid.values[r+dr][c+dc] = color
                placed = True
    
    # Fill remaining empty spaces with border color
    for r in range(rows):
        for c in range(6):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color
    
    return output_grid
