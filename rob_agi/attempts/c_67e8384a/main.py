from rob_agi.colored_grid import ColoredGrid

def solve_67e8384a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid by mirroring horizontally and vertically.
    
    The transformation process:
    1. Copy the input 3x3 grid to the top-left quadrant of a 6x6 grid
    2. Mirror the top-left quadrant horizontally to fill the top-right quadrant
    3. Mirror the entire top half vertically to fill the bottom half
    
    Args:
    input_grid (ColoredGrid): A 3x3 input grid

    Returns:
    ColoredGrid: A 6x6 output grid with the applied transformation
    """
    # Create a 6x6 output grid filled with zeros
    output = [[0 for _ in range(6)] for _ in range(6)]
    
    # Copy the input 3x3 grid to the top-left quadrant
    for i in range(3):
        for j in range(3):
            output[i][j] = input_grid.get_cell(i, j)
    
    # Mirror the top-left quadrant horizontally to fill the top-right quadrant
    for i in range(3):
        for j in range(3, 6):
            output[i][j] = output[i][5-j]
    
    # Mirror the entire top half vertically to fill the bottom half
    for i in range(3, 6):
        for j in range(6):
            output[i][j] = output[5-i][j]
    
    # Create and return a new ColoredGrid with the output values
    return ColoredGrid(values=output)
