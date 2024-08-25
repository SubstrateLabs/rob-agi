from rob_agi.colored_grid import ColoredGrid

def solve_9172f3a0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 9x9 output grid by expanding each cell
    into a 3x3 block of the same color.

    The function works as follows:
    1. Each cell in the 3x3 input grid is replicated into a 3x3 block in the output grid.
    2. The 9x9 output grid is created by mapping each cell to its corresponding
       cell in the input grid using integer division.

    Args:
    input_grid (ColoredGrid): A 3x3 ColoredGrid object representing the input.

    Returns:
    ColoredGrid: A 9x9 ColoredGrid object representing the expanded output.
    """
    input_values = input_grid.values
    
    output_values = [
        [input_values[i//3][j//3] for j in range(9)]
        for i in range(9)
    ]
    
    return ColoredGrid(values=output_values)
