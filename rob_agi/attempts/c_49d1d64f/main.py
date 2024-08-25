from rob_agi.colored_grid import ColoredGrid

def solve_49d1d64f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by expanding it and adding a border.
    
    The transformation follows these steps:
    1. Each cell in the input is replicated in a 2x2 pattern in the output.
    2. A border of zeros (black) is added around the expanded grid.
    3. The first and last columns of the inner expanded grid replicate the first and last columns of the input.
    4. The top and bottom rows of the inner expanded grid replicate the first and last rows of the input.
    5. The corners of the inner expanded grid are filled with the corresponding corner values from the input.
    
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
        output_values[i][1] = input_values[i//2][0]
        output_values[i][-2] = input_values[i//2][-1]
    
    # Handle top and bottom rows
    for j in range(1, 2*m+1):
        output_values[1][j] = input_values[0][j//2]
        output_values[-2][j] = input_values[-1][j//2]
    
    # Handle corners
    output_values[1][1] = input_values[0][0]
    output_values[1][-2] = input_values[0][-1]
    output_values[-2][1] = input_values[-1][0]
    output_values[-2][-2] = input_values[-1][-1]
    
    return ColoredGrid(values=output_values)
