from rob_agi.colored_grid import ColoredGrid

def solve_8ee62060(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by rotating the diagonal pattern from top-left to bottom-right.
    
    This function takes an input ColoredGrid and returns a new ColoredGrid where:
    1. Non-zero elements are moved to maintain their relative positions.
    2. The pattern is rotated to run from the bottom-left corner to the top-right corner.
    3. The overall dimensions of the grid remain unchanged.
    
    The transformation is achieved by:
    - Calculating each non-zero element's distance from the bottom-left corner.
    - Using these distances to determine new positions, effectively rotating the pattern.
    """
    # Get dimensions of the input grid
    rows, cols = input_grid.get_dimensions()
    
    # Create a new grid with the same dimensions, initialized with zeros
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Identify and transform non-zero elements
    for row in range(rows):
        for col in range(cols):
            if input_grid.values[row][col] != 0:
                # Calculate distances from bottom-left corner
                distance_row = (rows - 1) - row
                distance_col = col
                
                # Calculate new position
                new_row = (rows - 1) - distance_col
                new_col = distance_row
                
                # Transfer the element to its new position
                new_grid.values[new_row][new_col] = input_grid.values[row][col]
    
    return new_grid
