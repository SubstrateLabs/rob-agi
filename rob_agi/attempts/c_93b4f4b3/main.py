from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and rearranging internal shapes.
    
    1. Extracts the border from the left 6 columns of the input grid.
    2. Identifies and extracts shapes from the right side of the input grid.
    3. Inverts the extracted shapes vertically.
    4. Identifies empty spaces within the border shape.
    5. Places inverted shapes in empty spaces, starting from the top and maintaining vertical order.
    6. Fills any remaining empty space with the border color.
    
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
        inverted_shape = [((rows - 1 - r, c - 6), color) for r, c in shape]
        processed_shapes.append((inverted_shape, height, width))
    
    # Sort shapes by their original vertical position (top to bottom)
    processed_shapes.sort(key=lambda x: min(r for (r, _), _ in x[0]))
    
    # Identify empty spaces
    empty_spaces = []
    for r in range(rows):
        space_start = None
        for c in range(1, 5):
            if output_grid.values[r][c] == 0:
                if space_start is None:
                    space_start = c
            elif space_start is not None:
                empty_spaces.append((r, space_start, c - space_start))
                space_start = None
        if space_start is not None:
            empty_spaces.append((r, space_start, 5 - space_start))
    
    # Place shapes in empty spaces
    for shape, height, width in processed_shapes:
        placed = False
        for r in range(rows - height + 1):
            if placed:
                break
            for c in range(1, 5 - width + 1):
                if all(output_grid.values[r+dr][c+dc] == 0 for dr in range(height) for dc in range(width)):
                    for (sr, sc), color in shape:
                        if 0 <= r + (sr % height) < rows and 0 <= c + sc < 6:
                            output_grid.values[r + (sr % height)][c + sc] = color
                    placed = True
                    break
    
    # Fill remaining empty spaces with border color
    for r in range(rows):
        for c in range(6):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color
    
    return output_grid
