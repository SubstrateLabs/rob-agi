from rob_agi.colored_grid import ColoredGrid

def solve_b0722778(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by pairing non-black rows and extracting specific columns.
    
    The function creates a new grid with:
    - The same number of rows as the input grid
    - Always 2 columns
    - Non-black rows are paired:
      - For the upper row of each pair, the second column is taken from the rightmost non-black section
      - For the lower row of each pair, the first column is taken from the rightmost non-black section
    - Black (0) rows in the input are preserved as black rows in the output
    - If there's an odd number of non-black rows, the last row is treated as both upper and lower
    
    This effectively ignores the black separator columns and focuses on pairing the non-black rows.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Identify the last non-black section
    last_section_start = cols - 1
    while last_section_start > 0 and all(input_grid.values[r][last_section_start] == 0 for r in range(rows)):
        last_section_start -= 1
    col1, col2 = last_section_start - 1, last_section_start
    
    output_rows = []
    non_black_counter = 0
    
    for row in range(rows):
        if all(input_grid.values[row][c] == 0 for c in range(cols)):
            output_rows.append([0, 0])
        else:
            if non_black_counter % 2 == 0:  # Lower row of a pair
                output_rows.append([input_grid.values[row][col1], 0])
            else:  # Upper row of a pair
                output_rows[-1][1] = input_grid.values[row][col2]
            non_black_counter += 1
    
    # Handle odd number of non-black rows
    if non_black_counter % 2 != 0:
        last_non_black = next(i for i in range(len(output_rows) - 1, -1, -1) if output_rows[i] != [0, 0])
        output_rows[last_non_black][1] = input_grid.values[last_non_black][col2]
    
    return ColoredGrid(values=output_rows)
