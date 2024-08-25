from rob_agi.colored_grid import ColoredGrid

def solve_b0722778(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting specific columns for each non-black row.
    
    The function creates a new grid with:
    - The same number of rows as the input grid
    - Always 2 columns
    - For each non-black row:
      - The first column is taken from the rightmost non-black column of the input
      - The second column is taken from the second-rightmost non-black column of the input
    - Black (0) rows in the input are preserved as black rows in the output
    
    This effectively ignores the black separator columns and focuses on the rightmost non-black columns.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Identify the two rightmost non-black columns
    col1, col2 = cols - 1, cols - 2
    while col1 > 0 and all(input_grid.values[r][col1] == 0 for r in range(rows)):
        col1 -= 1
    while col2 >= 0 and all(input_grid.values[r][col2] == 0 for r in range(rows)):
        col2 -= 1
    
    output_rows = []
    
    for row in range(rows):
        if all(input_grid.values[row][c] == 0 for c in range(cols)):
            output_rows.append([0, 0])
        else:
            output_rows.append([input_grid.values[row][col1], input_grid.values[row][col2]])
    
    return ColoredGrid(values=output_rows)
