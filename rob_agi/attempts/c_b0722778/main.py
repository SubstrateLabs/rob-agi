from rob_agi.colored_grid import ColoredGrid

def solve_b0722778(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the rightmost two non-black columns for the entire grid.
    
    The function creates a new grid with:
    - The same number of rows as the input grid
    - Always 2 columns
    - For each row:
      - If the row is entirely black (0), it remains [0, 0] in the output
      - Otherwise, it takes values from the two rightmost non-black columns of the entire input grid
    
    This effectively ignores the black separator columns and focuses on the rightmost non-black columns
    of the entire grid, not just each individual row.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Identify the two rightmost non-black columns for the entire grid
    col1, col2 = -1, -1
    for c in range(cols - 1, -1, -1):
        if any(input_grid.values[r][c] != 0 for r in range(rows)):
            if col1 == -1:
                col1 = c
            elif col2 == -1:
                col2 = c
                break
    
    output_rows = []
    
    for row in range(rows):
        if all(input_grid.values[row][c] == 0 for c in range(cols)):
            output_rows.append([0, 0])
        else:
            output_rows.append([input_grid.values[row][col1], input_grid.values[row][col2]])
    
    return ColoredGrid(values=output_rows)
