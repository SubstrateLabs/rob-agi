from rob_agi.colored_grid import ColoredGrid

def solve_f0afb749(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by:
    1. Doubling the grid size
    2. Expanding non-black squares into 2x2 blocks
    3. Adding blue squares (1s) diagonally from top-left to bottom-right
       in empty rows and columns until a non-empty row or column is encountered
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

    # Helper functions
    def is_row_empty(row):
        return all(cell == 0 for cell in output_grid[row])

    def is_column_empty(col):
        return all(row[col] == 0 for row in output_grid)

    # Step 3: Place blue squares diagonally
    row, col = 0, 0
    while row < output_rows and col < output_cols:
        if is_row_empty(row) and is_column_empty(col):
            output_grid[row][col] = 1  # Place blue square
            row += 1
            col += 1
        else:
            break  # Stop if we hit a non-empty row or column

    # Step 4: Create and return the final ColoredGrid
    return ColoredGrid(values=output_grid)
