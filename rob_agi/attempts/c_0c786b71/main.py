from rob_agi.colored_grid import ColoredGrid

def solve_0c786b71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 3x4 grid into a larger 6x8 grid using a specific pattern.
    
    The transformation follows these steps:
    1. Fill the top-left 3x4 quadrant with the input rows in this order:
       - Last row: swap first and last elements, keep middle elements in place
       - First row: swap first and last elements, swap middle elements
       - Second row: swap first and last elements, keep middle elements in place
    2. Mirror the top-left quadrant horizontally to fill the top-right quadrant.
    3. Mirror the entire top half vertically to create the bottom half.
    
    This creates a symmetrical expansion of the input grid with specific row reordering and element swapping.
    """
    # Initialize 6x8 output grid
    output = [[0 for _ in range(8)] for _ in range(6)]
    
    # Fill top-left quadrant
    # Last input row (becomes first output row)
    output[0][0] = input_grid.values[2][3]
    output[0][1] = input_grid.values[2][1]
    output[0][2] = input_grid.values[2][2]
    output[0][3] = input_grid.values[2][0]
    
    # First input row (becomes second output row)
    output[1][0] = input_grid.values[0][3]
    output[1][1] = input_grid.values[0][2]
    output[1][2] = input_grid.values[0][1]
    output[1][3] = input_grid.values[0][0]
    
    # Second input row (becomes third output row)
    output[2][0] = input_grid.values[1][3]
    output[2][1] = input_grid.values[1][1]
    output[2][2] = input_grid.values[1][2]
    output[2][3] = input_grid.values[1][0]
    
    # Mirror top-left quadrant horizontally
    for row in range(3):
        for col in range(4):
            output[row][7-col] = output[row][col]
    
    # Mirror top half vertically
    for row in range(3):
        output[5-row] = output[row].copy()
    
    return ColoredGrid(values=output)
