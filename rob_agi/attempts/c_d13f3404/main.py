from rob_agi.colored_grid import ColoredGrid

def solve_d13f3404(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid by propagating non-zero elements diagonally.
    
    The transformation follows these rules:
    1. The input 3x3 grid is expanded to a 6x6 output grid.
    2. The first row and column of the output grid match the first row and column of the input grid.
    3. Non-zero elements from the input grid propagate diagonally down-right in the output grid.
    4. When a diagonal reaches the right edge, it wraps around to the left side of the next row.
    5. This pattern continues until the entire 6x6 grid is filled.
    6. The propagation only fills empty (0) cells, preserving existing non-zero values.
    7. The propagation starts from the top-left corner and proceeds row by row, column by column.
    
    Args:
    input_grid (ColoredGrid): The input 3x3 grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed 6x6 grid.
    """
    input_values = input_grid.values
    output_values = [[0 for _ in range(6)] for _ in range(6)]
    
    # Copy the first row and column from input to output
    for i in range(3):
        output_values[0][i] = input_values[0][i]
        output_values[i][0] = input_values[i][0]
    
    def propagate(value, start_row, start_col):
        for i in range(6):
            row = (start_row + i) % 6
            col = (start_col + i) % 6
            if output_values[row][col] == 0:  # Only fill if the cell is empty
                output_values[row][col] = value
    
    # Propagate non-zero elements starting from the top-left corner
    for i in range(3):
        for j in range(3):
            if input_values[i][j] != 0:
                propagate(input_values[i][j], i, j)
    
    return ColoredGrid(values=output_values)
