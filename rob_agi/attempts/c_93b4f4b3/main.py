from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and rearranging internal shapes.
    
    1. Extracts the border and internal shapes from the input grid.
    2. Creates a new 6-column grid with the same number of rows as the input.
    3. Places the border in the output grid.
    4. Sorts internal shapes based on their lowest row, in descending order.
    5. Places sorted shapes within the border, centered horizontally and inverted vertically.
    6. Fills remaining space with the border color.
    
    Returns the transformed ColoredGrid.
    """
    rows, cols = input_grid.get_dimensions()
    border_color = input_grid.values[0][0]
    
    # Extract border
    border = [(r, c) for r in range(rows) for c in range(6) if input_grid.values[r][c] == border_color]
    
    # Extract internal shapes
    internal_shapes = []
    for color in range(10):
        if color != border_color:
            shapes = input_grid.find_connected_regions(color)
            for shape in shapes:
                if any(c >= 6 for _, c in shape):  # Only consider shapes from the right side
                    lowest_row = max(r for r, _ in shape)
                    internal_shapes.append((shape, color, lowest_row))
    
    # Sort shapes by their lowest row, in descending order
    internal_shapes.sort(key=lambda x: x[2], reverse=True)
    
    # Create output grid
    output_grid = ColoredGrid(values=[[border_color for _ in range(6)] for _ in range(rows)])
    
    # Place border
    for r, c in border:
        output_grid.values[r][c] = border_color
    
    # Place shapes
    current_row = 1  # Start placing shapes from the top
    for shape, color, _ in internal_shapes:
        shape_height = max(r for r, _ in shape) - min(r for r, _ in shape) + 1
        shape_width = max(c for _, c in shape) - min(c for _, c in shape) + 1
        
        # Find space for the shape
        while current_row + shape_height < rows and any(output_grid.values[current_row][c] != border_color for c in range(1, 5)):
            current_row += 1
        
        if current_row + shape_height >= rows:
            raise ValueError("Unable to place all shapes")
        
        # Center the shape horizontally
        left = max(1, (6 - shape_width) // 2)
        
        # Place shape
        min_r = min(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        for r, c in shape:
            output_grid.values[current_row + shape_height - 1 - (r - min_r)][left + c - min_c] = color
        
        current_row += shape_height + 1  # Move to the next row after this shape
    
    # Fill remaining space with border color
    for r in range(rows):
        for c in range(1, 5):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color
    
    return output_grid
