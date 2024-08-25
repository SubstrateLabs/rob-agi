from rob_agi.colored_grid import ColoredGrid

def solve_363442ee(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 363442ee challenge by replicating a 3x3 pattern from the top-left corner
    of the input grid to the right of a separator column (filled with 5s).
    
    1. Extract the 3x3 pattern from the top-left corner.
    2. Identify the separator column (filled with 5s).
    3. Create a deep copy of the input grid.
    4. Process the grid in 3x3 blocks to the right of the separator:
       - Always replicate the pattern in the first column after the separator.
       - For subsequent columns, replicate only if a blue square (1) is present in that specific column of the input grid.
       - When replicating, consider the vertical position:
         * In the first row of 3x3 blocks, always replicate.
         * In subsequent rows, replicate only if there's a blue square (1) anywhere in that row of the input grid.
    5. Return the modified grid.
    """
    def has_one_in_row(grid: list[list[int]], row: int) -> bool:
        return any(grid[row][col] == 1 for col in range(len(grid[0])))

    def has_one_in_column(grid: list[list[int]], col: int) -> bool:
        return any(grid[row][col] == 1 for row in range(len(grid)))

    def replicate_pattern(grid: list[list[int]], pattern: list[list[int]], start_row: int, start_col: int):
        for i in range(3):
            for j in range(3):
                if start_row + i < len(grid) and start_col + j < len(grid[0]):
                    grid[start_row + i][start_col + j] = pattern[i][j]

    # Extract the 3x3 pattern from the top-left corner
    pattern = [input_grid.values[i][:3] for i in range(3)]

    # Identify the separator column (the column filled with 5s)
    separator_col = next(col for col in range(len(input_grid.values[0])) if all(row[col] == 5 for row in input_grid.values))

    # Create a deep copy of the input grid to modify
    output = input_grid.deep_copy()

    # Process the grid in 3x3 blocks, focusing on the area to the right of the separator
    for row in range(0, len(output.values), 3):
        first_col_after_separator = True
        for col in range(separator_col + 1, len(output.values[0]), 3):
            if row == 0 or has_one_in_row(input_grid.values, row):
                if first_col_after_separator or has_one_in_column(input_grid.values, col):
                    replicate_pattern(output.values, pattern, row, col)
            first_col_after_separator = False

    return output
