from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and rearranging internal shapes.
    
    1. Extracts the border from the left 6 columns of the input grid.
    2. Identifies and extracts shapes from the right side of the input grid.
    3. Inverts the extracted shapes vertically and sorts them by area.
    4. Identifies empty spaces within the border shape.
    5. Places inverted shapes in empty spaces, prioritizing larger shapes and spaces.
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
        if color != border_color:
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
        inverted_shape = [((max_r - (r - min_r), c - min_c), color) for r, c in shape]
        processed_shapes.append((inverted_shape, height, width, height * width))
    
    # Sort shapes by area in descending order
    processed_shapes.sort(key=lambda x: x[3], reverse=True)
    
    # Identify empty spaces
    empty_spaces = []
    for r in range(rows):
        space_start = None
        for c in range(1, 5):
            if output_grid.values[r][c] != border_color:
                if space_start is None:
                    space_start = c
            elif space_start is not None:
                empty_spaces.append((r, space_start, c - space_start))
                space_start = None
        if space_start is not None:
            empty_spaces.append((r, space_start, 5 - space_start))
    
    # Merge vertically adjacent empty spaces
    merged_spaces = []
    for r, c, width in sorted(empty_spaces):
        if merged_spaces and merged_spaces[-1][1] == c and merged_spaces[-1][2] == width and merged_spaces[-1][0] + merged_spaces[-1][3] == r:
            merged_spaces[-1] = (merged_spaces[-1][0], c, width, merged_spaces[-1][3] + 1)
        else:
            merged_spaces.append((r, c, width, 1))
    
    # Sort empty spaces by area (height * width) in descending order
    merged_spaces.sort(key=lambda x: x[2] * x[3], reverse=True)
    
    # Place shapes in empty spaces
    for space in merged_spaces:
        space_r, space_c, space_width, space_height = space
        for i, (shape, height, width, _) in enumerate(processed_shapes):
            if height <= space_height and width <= space_width:
                # Place the shape
                for (r, c), color in shape:
                    if 0 <= space_r + r < rows and 0 <= space_c + c < 6:
                        output_grid.values[space_r + r][space_c + c] = color
                processed_shapes.pop(i)
                break
    
    # Fill remaining empty spaces with border color
    for r in range(rows):
        for c in range(6):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color
    
    return output_grid
