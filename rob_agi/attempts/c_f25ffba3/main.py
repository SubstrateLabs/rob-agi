from rob_agi.colored_grid import ColoredGrid

def solve_f25ffba3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by flipping the top half vertically.
    
    The transformation rule:
    1. The bottom 5 rows remain unchanged.
    2. The top 5 rows are a vertical flip of the bottom 5 rows.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    # Get the grid values
    grid = input_grid.values
    
    # Extract the bottom 5 rows
    bottom_half = grid[5:]
    
    # Create the top half by reversing the order of the bottom half
    top_half = bottom_half[::-1]
    
    # Combine the flipped top half with the original bottom half
    result = top_half + bottom_half
    
    # Return a new ColoredGrid with the transformed values
    return ColoredGrid(values=result)
