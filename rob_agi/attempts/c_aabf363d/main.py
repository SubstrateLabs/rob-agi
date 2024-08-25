from rob_agi.colored_grid import ColoredGrid

def solve_aabf363d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by replacing all non-zero, non-anchor colors with the anchor color.
    The anchor color is the color in the bottom-left corner of the input grid.
    The bottom-left corner is set to 0 in the output grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    # Get the dimensions of the input grid
    height, width = input_grid.get_dimensions()
    
    # Get the anchor color from the bottom-left corner
    anchor_color = input_grid.get_cell(height - 1, 0)
    
    # Create a new grid for the output
    output = input_grid.deep_copy()
    
    # Iterate through the grid and apply the transformation
    for row in range(height):
        for col in range(width):
            cell_value = input_grid.get_cell(row, col)
            if cell_value != 0 and cell_value != anchor_color:
                output.set_cell(row, col, anchor_color)
    
    # Set the bottom-left corner to 0
    output.set_cell(height - 1, 0, 0)
    
    return output
