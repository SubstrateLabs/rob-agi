from rob_agi.colored_grid import ColoredGrid

def solve_b0722778(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the last two columns of the rightmost non-black section.
    
    The function creates a new grid with:
    - The same number of rows as the input grid
    - Always 2 columns
    - The output columns correspond to the last two columns of the rightmost non-black section
    - Black (0) rows in the input are preserved as black rows in the output
    
    This effectively ignores the black separator columns and focuses on the last meaningful section.
    """
    # Get the dimensions of the input grid
    rows, cols = input_grid.get_dimensions()
    
    # Identify the last non-black section
    last_section_start = cols - 1
    while last_section_start > 0 and all(input_grid.values[r][last_section_start] == 0 for r in range(rows)):
        last_section_start -= 1
    col1, col2 = last_section_start - 1, last_section_start
    
    # Initialize an empty list for output rows
    output_rows = []
    
    # Process each row of the input grid
    for row in range(rows):
        if all(input_grid.values[row][c] == 0 for c in range(cols)):
            # If the row is all black, preserve it as a black row
            output_rows.append([0, 0])
        else:
            # Extract the values from the last two columns of the non-black section
            output_rows.append([input_grid.values[row][col1], input_grid.values[row][col2]])
    
    # Create a new ColoredGrid object using the list of output rows
    output_grid = ColoredGrid(values=output_rows)
    
    # Return the new ColoredGrid object
    return output_grid
