from rob_agi.colored_grid import ColoredGrid

def solve_62c24649(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid by:
    1. Replicating the input grid in all four corners of the output grid
    2. Mirroring the edges inwards to fill the center
    """
    # Create a 6x6 output grid initialized with zeros
    output = ColoredGrid(values=[[0 for _ in range(6)] for _ in range(6)])
    
    # Copy the 3x3 input grid to all four corners
    for i in range(3):
        for j in range(3):
            value = input_grid.get_cell(i, j)
            output.set_cell(i, j, value)
            output.set_cell(i, 5-j, value)
            output.set_cell(5-i, j, value)
            output.set_cell(5-i, 5-j, value)
    
    # Mirror the edges inwards to fill the center
    for i in range(3):
        output.set_cell(i, 2, output.get_cell(i, 1))
        output.set_cell(i, 3, output.get_cell(i, 4))
        output.set_cell(5-i, 2, output.get_cell(5-i, 1))
        output.set_cell(5-i, 3, output.get_cell(5-i, 4))
    
    for j in range(6):
        output.set_cell(2, j, output.get_cell(1, j))
        output.set_cell(3, j, output.get_cell(4, j))
    
    return output
