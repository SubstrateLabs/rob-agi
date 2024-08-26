from rob_agi.colored_grid import ColoredGrid

def solve_8ee62060(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by rotating the diagonal pattern.
    
    This function takes an input ColoredGrid and returns a new ColoredGrid where:
    1. Non-zero elements are moved to maintain their relative positions.
    2. The pattern is rotated 90 degrees clockwise around the center of the grid.
    3. The overall dimensions of the grid remain unchanged.
    
    The transformation is achieved by:
    - Calculating each non-zero element's position relative to the grid center.
    - Rotating this position 90 degrees clockwise.
    - Placing the element in its new rotated position.
    """
    # Get dimensions of the input grid
    rows, cols = input_grid.get_dimensions()
    
    # Create a new grid with the same dimensions, initialized with zeros
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Calculate the center of the grid
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2
    
    # Identify and transform non-zero elements
    for row in range(rows):
        for col in range(cols):
            if input_grid.values[row][col] != 0:
                # Calculate position relative to center
                rel_row = row - center_row
                rel_col = col - center_col
                
                # Rotate 90 degrees clockwise
                new_rel_row = rel_col
                new_rel_col = -rel_row
                
                # Calculate new absolute position
                new_row = int(center_row + new_rel_row)
                new_col = int(center_col + new_rel_col)
                
                # Transfer the element to its new position
                new_grid.values[new_row][new_col] = input_grid.values[row][col]
    
    return new_grid
