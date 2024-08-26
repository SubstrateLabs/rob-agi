from rob_agi.colored_grid import ColoredGrid

def solve_f0afb749(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Doubling the grid size
    2. Expanding non-black squares into 2x2 blocks
    3. Adding blue squares (1s) in empty rows and columns:
       - In empty rows: leftmost and rightmost empty columns
       - In empty columns: topmost and bottommost empty rows
    """
    # Step 1: Initialize the output grid
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

    # Step 2: Expand non-black squares
    for r in range(input_rows):
        for c in range(input_cols):
            if input_grid.values[r][c] != 0:
                color = input_grid.values[r][c]
                output_grid[r*2][c*2] = color
                output_grid[r*2][c*2+1] = color
                output_grid[r*2+1][c*2] = color
                output_grid[r*2+1][c*2+1] = color

    # Step 3: Identify empty rows and columns
    empty_rows = [all(cell == 0 for cell in row) for row in output_grid]
    empty_cols = [all(row[c] == 0 for row in output_grid) for c in range(output_cols)]

    # Step 4: Place blue squares in empty rows
    for r in range(output_rows):
        if empty_rows[r]:
            left_col = next((c for c in range(output_cols) if empty_cols[c]), None)
            right_col = next((c for c in range(output_cols-1, -1, -1) if empty_cols[c]), None)
            if left_col is not None:
                output_grid[r][left_col] = 1
            if right_col is not None and right_col != left_col:
                output_grid[r][right_col] = 1

    # Step 5: Place blue squares in empty columns
    for c in range(output_cols):
        if empty_cols[c]:
            top_row = next((r for r in range(output_rows) if empty_rows[r]), None)
            bottom_row = next((r for r in range(output_rows-1, -1, -1) if empty_rows[r]), None)
            if top_row is not None:
                output_grid[top_row][c] = 1
            if bottom_row is not None and bottom_row != top_row:
                output_grid[bottom_row][c] = 1

    # Step 6: Create and return the final ColoredGrid
    return ColoredGrid(values=output_grid)
