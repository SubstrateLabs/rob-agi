from rob_agi.colored_grid import ColoredGrid

def solve_2281f1f4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. The first row remains unchanged
    2. The last column remains unchanged
    3. For rows with a 5 in the last column (except the first row),
       fill with 2s in columns that have a 5 in the first row (excluding the last column)
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid
    """
    # Create a deep copy of the input grid
    output = input_grid.deep_copy()
    
    # Get grid dimensions
    rows, cols = input_grid.get_dimensions()
    
    # Identify columns with 5s in the first row (excluding the last column)
    columns_with_5 = [col for col in range(cols - 1) if input_grid.get_cell(0, col) == 5]
    
    # Iterate through rows (starting from the second row)
    for row in range(1, rows):
        # Check if the last column has a 5
        if input_grid.get_cell(row, cols - 1) == 5:
            # Fill the row with 2s in the identified columns
            for col in columns_with_5:
                output.set_cell(row, col, 2)
    
    return output
