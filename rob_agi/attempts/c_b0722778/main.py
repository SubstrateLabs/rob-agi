from rob_agi.colored_grid import ColoredGrid

def solve_b0722778(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the 2nd and 3rd columns of the first 3-column section.
    
    The function creates a new grid with:
    - The same number of rows as the input grid
    - Always 2 columns
    - The first output column corresponds to the 2nd column of the input
    - The second output column corresponds to the 3rd column of the input
    
    This effectively ignores the black separator columns and other sections in the input grid.
    """
    # Get the dimensions of the input grid
    rows, _ = input_grid.get_dimensions()
    
    # Initialize an empty list for output rows
    output_rows = []
    
    # Iterate through each row of the input grid
    for row in range(rows):
        # Create a new list for the current output row
        output_row = [
            # Add the color from the 2nd column (index 1)
            input_grid.values[row][1],
            # Add the color from the 3rd column (index 2)
            input_grid.values[row][2]
        ]
        # Append this output row to the list of output rows
        output_rows.append(output_row)
    
    # Create a new ColoredGrid object using the list of output rows
    output_grid = ColoredGrid(values=output_rows)
    
    # Return the new ColoredGrid object
    return output_grid
