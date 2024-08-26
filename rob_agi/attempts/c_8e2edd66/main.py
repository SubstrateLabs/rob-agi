from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell and creating connections.
    
    The transformation follows these rules:
    1. Map each non-zero value from the input grid to a corresponding position in the output grid.
    2. Create connections between adjacent non-zero values in the input grid.
    3. Fill the spaces between connected values to form patterns.
    4. All other positions in the output grid remain zero (black).

    This creates a pattern that preserves the structure and connections of the input while expanding it into a larger grid.
    """
    # Create a new 9x9 grid filled with zeros
    output_values = [[0 for _ in range(9)] for _ in range(9)]
    
    # Map input values to output grid
    for i in range(3):
        for j in range(3):
            v = input_grid.values[i][j]
            if v != 0:
                output_values[i*3][j*3] = v

    # Create connections between adjacent non-zero values
    for i in range(3):
        for j in range(3):
            v = input_grid.values[i][j]
            if v != 0:
                # Right neighbor
                if j < 2 and input_grid.values[i][j+1] != 0:
                    output_values[i*3][j*3+1] = output_values[i*3][j*3+2] = v
                # Bottom neighbor
                if i < 2 and input_grid.values[i+1][j] != 0:
                    output_values[i*3+1][j*3] = output_values[i*3+2][j*3] = v
                # Bottom-right neighbor
                if i < 2 and j < 2 and input_grid.values[i+1][j+1] != 0:
                    output_values[i*3+1][j*3] = output_values[i*3+2][j*3] = output_values[i*3+3][j*3] = v
                    output_values[i*3+3][j*3+1] = output_values[i*3+3][j*3+2] = v
                    output_values[i*3][j*3+1] = output_values[i*3][j*3+2] = output_values[i*3][j*3+3] = v
                    output_values[i*3+1][j*3+3] = output_values[i*3+2][j*3+3] = v
    
    # Create and return a new ColoredGrid with the output values
    return ColoredGrid(values=output_values)
