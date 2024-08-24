from rob_agi.colored_grid import ColoredGrid

def solve_49d1d64f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a border.
    
    The transformation follows these steps:
    1. Each cell in the input is replicated in a 2x2 pattern in the output.
    2. A border of zeros (black) is added around the expanded grid.
    3. The first and last columns of the inner expanded grid replicate the first and last columns of the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    input_values = input_grid.values
    n, m = len(input_values), len(input_values[0])
    output_values = [[0 for _ in range(2*m+2)] for _ in range(2*n+2)]
    
    # Replicate input cells in 2x2 pattern
    for i in range(n):
        for j in range(m):
            val = input_values[i][j]
            output_values[2*i+1][2*j+1] = val
            output_values[2*i+1][2*j+2] = val
            output_values[2*i+2][2*j+1] = val
            output_values[2*i+2][2*j+2] = val
    
    # Handle edge columns
    for i in range(1, 2*n+1):
        output_values[i][1] = output_values[i][2]
        output_values[i][-2] = output_values[i][-3]
    
    return ColoredGrid(values=output_values)
