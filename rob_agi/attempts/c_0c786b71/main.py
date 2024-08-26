from rob_agi.colored_grid import ColoredGrid

def solve_0c786b71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 3x4 grid into a larger 6x8 grid using a specific pattern.
    
    The transformation follows these steps:
    1. Fill the top-left 3x4 quadrant with the input rows in a specific order:
       last row, first row, second row of the input.
    2. Mirror the top-left quadrant horizontally to fill the top-right quadrant.
    3. Mirror the entire top half vertically to create the bottom half.
    
    This creates a symmetrical expansion of the input grid with specific row reordering.
    """
    # Initialize 6x8 output grid
    output = [[0 for _ in range(8)] for _ in range(6)]
    
    # Fill top-left quadrant
    output[0] = input_grid.values[2][:4]  # Last row of input
    output[1] = input_grid.values[0][:4]  # First row of input
    output[2] = input_grid.values[1][:4]  # Second row of input
    
    # Mirror top-left quadrant horizontally
    for row in range(3):
        for col in range(4):
            output[row][7-col] = output[row][col]
    
    # Mirror top half vertically
    for row in range(3):
        output[5-row] = output[row].copy()
    
    return ColoredGrid(values=output)
