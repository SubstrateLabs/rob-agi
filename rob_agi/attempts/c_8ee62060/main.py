from rob_agi.colored_grid import ColoredGrid

def solve_8ee62060(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by moving 2x2 subgrids to their opposite corners.
    
    This function takes an input ColoredGrid and returns a new ColoredGrid where:
    1. The grid is divided into 2x2 subgrids.
    2. Each non-zero 2x2 subgrid is moved to its opposite corner in the grid.
    3. The internal structure of each 2x2 subgrid is preserved.
    4. The overall dimensions of the grid remain unchanged.
    
    This transformation effectively reverses any diagonal patterns in the grid
    while maintaining the relative positions of elements within each 2x2 subgrid.
    """
    # Get dimensions of the input grid
    rows, cols = input_grid.get_dimensions()
    
    # Calculate the number of 2x2 subgrids in each dimension
    subgrid_rows = rows // 2
    subgrid_cols = cols // 2
    
    # Create a new grid with the same dimensions, initialized with zeros
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Iterate through 2x2 subgrids
    for i in range(subgrid_rows):
        for j in range(subgrid_cols):
            # Check if the subgrid contains any non-zero values
            if any(input_grid.values[2*i+x][2*j+y] != 0 for x in range(2) for y in range(2)):
                # Calculate new position for the subgrid
                new_i = subgrid_rows - 1 - i
                new_j = subgrid_cols - 1 - j
                
                # Transfer the 2x2 subgrid to its new position
                for x in range(2):
                    for y in range(2):
                        new_grid.values[2*new_i+x][2*new_j+y] = input_grid.values[2*i+x][2*j+y]
    
    return new_grid
