from rob_agi.colored_grid import ColoredGrid

def solve_0c786b71(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid by mirroring it in all directions.
    
    The function doubles the size of the input grid, creating a 6x8 output grid:
    1. Copy the input grid to the top-left quadrant.
    2. Mirror horizontally for the top-right quadrant.
    3. Mirror vertically for the bottom-left quadrant.
    4. Mirror both horizontally and vertically for the bottom-right quadrant.
    
    This creates a symmetrical expansion of the input grid.
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
        for j in range(input_cols, output_cols):
            output_values[i][j] = input_grid.values[i][output_cols - 1 - j]
    
    # Fill bottom-left quadrant (vertical mirror)
    for i in range(input_rows, output_rows):
        for j in range(input_cols):
            output_values[i][j] = input_grid.values[output_rows - 1 - i][j]
    
    # Fill bottom-right quadrant (both horizontal and vertical mirror)
    for i in range(input_rows, output_rows):
        for j in range(input_cols, output_cols):
            output_values[i][j] = input_grid.values[output_rows - 1 - i][output_cols - 1 - j]
    
    return ColoredGrid(values=output_values)
