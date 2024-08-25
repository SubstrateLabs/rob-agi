from rob_agi.colored_grid import ColoredGrid

def solve_ccd554ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the ccd554ac challenge by expanding the input grid into a square grid.
    
    The solution works as follows:
    1. Determine the expansion factor based on the larger dimension of the input grid.
    2. Create a new grid that is square, with dimensions (expansion_factor * input_rows) x (expansion_factor * input_cols).
    3. Fill the new grid by tiling the input grid pattern, using modulo operations to map new cells to the original grid.
    
    This approach ensures that the output is always square and each dimension is a multiple of the input dimensions,
    effectively tiling the original pattern to fill the larger grid.
    """
    # Analyze the input grid
    original_rows, original_cols = input_grid.get_dimensions()
    
    # Calculate the expansion factor
    expansion_factor = max(original_rows, original_cols)
    
    # Create the expanded grid
    new_rows = original_rows * expansion_factor
    new_cols = original_cols * expansion_factor
    new_grid = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
    
    # Fill the expanded grid
    for i in range(new_rows):
        for j in range(new_cols):
            original_row = i % original_rows
            original_col = j % original_cols
            new_grid[i][j] = input_grid.values[original_row][original_col]
    
    # Return the new expanded grid
    return ColoredGrid(values=new_grid)
