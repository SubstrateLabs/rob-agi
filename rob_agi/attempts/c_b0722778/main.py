from rob_agi.colored_grid import ColoredGrid

def solve_b0722778(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the two rightmost non-black columns for each section.
    
    The function creates a new grid with:
    - The same number of rows as the input grid
    - Always 2 columns
    - For each row:
      - If the row is entirely black (0), it remains [0, 0] in the output
      - Otherwise, it takes values from the two rightmost non-black columns of its section
    
    A section is defined as a group of rows separated by entirely black rows.
    This effectively ignores the black separator columns and rows, focusing on the
    rightmost non-black columns of each section independently.
    """
    rows, cols = input_grid.get_dimensions()
    
    output_rows = []
    section_start = 0
    
    for row in range(rows):
        if all(input_grid.values[row][c] == 0 for c in range(cols)):
            # Process the previous section
            if section_start < row:
                col1, col2 = find_rightmost_columns(input_grid, section_start, row)
                for r in range(section_start, row):
                    output_rows.append([input_grid.values[r][col1], input_grid.values[r][col2]])
            output_rows.append([0, 0])
            section_start = row + 1
    
    # Process the last section
    if section_start < rows:
        col1, col2 = find_rightmost_columns(input_grid, section_start, rows)
        for r in range(section_start, rows):
            output_rows.append([input_grid.values[r][col1], input_grid.values[r][col2]])
    
    return ColoredGrid(values=output_rows)

def find_rightmost_columns(input_grid: ColoredGrid, start_row: int, end_row: int) -> tuple[int, int]:
    """Find the two rightmost non-black columns in the given row range."""
    cols = input_grid.get_dimensions()[1]
    col1, col2 = -1, -1
    for c in range(cols - 1, -1, -1):
        if any(input_grid.values[r][c] != 0 for r in range(start_row, end_row)):
            if col1 == -1:
                col1 = c
            elif col2 == -1:
                col2 = c
                break
    return col1, col2
