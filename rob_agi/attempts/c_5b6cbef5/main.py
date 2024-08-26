from rob_agi.colored_grid import ColoredGrid

def solve_5b6cbef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into a 16x16 output grid by expanding the pattern.
    
    The solution involves the following steps:
    1. Create a new 16x16 grid filled with zeros (black)
    2. Copy the 4x4 input pattern to all four corners of the 16x16 grid
    3. Fill the edges between the corners by extending the pattern for 4 cells and leaving 4 cells empty
    4. Fill the center 8x8 area with extensions of the corner patterns
    5. Ensure symmetry in the center area
    
    This process creates a larger pattern that preserves the input's structure
    while expanding it across the 16x16 grid, maintaining symmetry and proper spacing.
    """
    # Initialize a new 16x16 ColoredGrid
    new_grid = ColoredGrid(values=[[0 for _ in range(16)] for _ in range(16)])
    
    # Copy input to corners
    for i in range(4):
        for j in range(4):
            value = input_grid.values[i][j]
            new_grid.values[i][j] = value
            new_grid.values[i][j+12] = value
            new_grid.values[i+12][j] = value
            new_grid.values[i+12][j+12] = value
    
    # Fill edge sections
    for i in range(4):
        for j in range(4, 8):
            new_grid.values[i][j] = input_grid.values[i][j-4]
            new_grid.values[i+12][j] = input_grid.values[i][j-4]
    for i in range(4, 8):
        for j in range(4):
            new_grid.values[i][j] = input_grid.values[i-4][j]
            new_grid.values[i][j+12] = input_grid.values[i-4][j]
    
    # Fill center section
    for i in range(4, 8):
        for j in range(4, 8):
            new_grid.values[i][j] = input_grid.values[i-2][j-2]
            new_grid.values[i][j+4] = input_grid.values[i-2][j-4]
            new_grid.values[i+4][j] = input_grid.values[i-4][j-2]
            new_grid.values[i+4][j+4] = input_grid.values[i-4][j-4]
    
    # Ensure symmetry in center section
    for i in range(4, 12):
        for j in range(4, 12):
            if new_grid.values[i][j] != 0:
                new_grid.values[15-i][j] = new_grid.values[i][j]
                new_grid.values[i][15-j] = new_grid.values[i][j]
                new_grid.values[15-i][15-j] = new_grid.values[i][j]
    
    return new_grid
