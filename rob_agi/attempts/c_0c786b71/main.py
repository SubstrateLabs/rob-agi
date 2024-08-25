from rob_agi.colored_grid import ColoredGrid

def solve_0c786b71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 3x4 grid into a larger 6x8 grid by mirroring it in all directions.
    
    The function creates a symmetrical expansion of the input grid:
    1. Copy the input grid to the top-left quadrant.
    2. Mirror horizontally for the top-right quadrant.
    3. Swap the first and third rows of the top half.
    4. Mirror the entire top half vertically to create the bottom half.
    
    This creates a symmetrical expansion of the input grid with specific row swaps.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    
    output_values = [[0 for _ in range(output_cols)] for _ in range(output_rows)]
    
    # Fill top-left quadrant (direct copy)
    for i in range(input_rows):
        for j in range(input_cols):
            output_values[i][j] = input_grid.values[i][j]
    
    # Fill top-right quadrant (horizontal mirror)
    for i in range(input_rows):
        for j in range(input_cols):
            output_values[i][j + input_cols] = input_grid.values[i][input_cols - 1 - j]
    
    # Swap first and third rows of the top half
    output_values[0], output_values[2] = output_values[2], output_values[0]
    
    # Mirror the entire top half vertically to create the bottom half
    for i in range(input_rows):
        for j in range(output_cols):
            output_values[i + input_rows][j] = output_values[input_rows - 1 - i][j]
    
    return ColoredGrid(values=output_values)
