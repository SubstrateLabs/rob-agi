from rob_agi.colored_grid import ColoredGrid

def solve_5ffb2104(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-empty columns to the right side of the grid.
    
    The transformation maintains the following properties:
    1. All non-empty columns are moved to the rightmost available positions.
    2. The vertical order of elements within each column is preserved.
    3. The relative horizontal order of non-empty columns is maintained.
    4. Empty columns are moved to the left side of the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with all non-empty columns moved to the right side.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Identify non-empty columns
    non_empty_cols = [col for col in range(cols) if any(input_grid.values[row][col] != 0 for row in range(rows))]
    
    # Calculate new positions
    new_start = cols - len(non_empty_cols)
    
    # Transfer data
    for i, col in enumerate(non_empty_cols):
        new_col = new_start + i
        for row in range(rows):
            new_grid[row][new_col] = input_grid.values[row][col]
    
    return ColoredGrid(values=new_grid)
