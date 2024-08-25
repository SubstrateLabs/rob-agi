from rob_agi.colored_grid import ColoredGrid

def solve_3af2c5a8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into an 8x6 grid by mirroring horizontally and vertically.
    
    The solution follows these steps:
    1. Create an 8x6 grid filled with zeros
    2. Copy the input grid to the top-left corner
    3. Mirror the left half horizontally to fill the right half
    4. Mirror the top half vertically to fill the bottom half
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed 8x6 grid
    """
    # Create an 8x6 grid filled with zeros
    output = ColoredGrid(values=[[0 for _ in range(8)] for _ in range(6)])
    
    # Copy the input grid to the top-left corner
    for i in range(len(input_grid.values)):
        for j in range(len(input_grid.values[0])):
            output.set_cell(i, j, input_grid.get_cell(i, j))
    
    # Mirror the left half horizontally to fill the right half
    for i in range(3):
        for j in range(4):
            output.set_cell(i, 7-j, output.get_cell(i, j))
    
    # Mirror the top half vertically to fill the bottom half
    for i in range(3):
        for j in range(8):
            output.set_cell(5-i, j, output.get_cell(i, j))
    
    return output
