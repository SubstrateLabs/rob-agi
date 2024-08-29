from rob_agi.colored_grid import ColoredGrid

def solve_8eb1be9a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 8eb1be9a challenge by replicating a 3-row pattern vertically from the top.
    
    This function identifies the first non-zero row in the input grid,
    extracts a 3-row pattern starting from that row, and then replicates
    this pattern vertically to fill the entire output grid. The pattern
    always starts from the top of the output grid, regardless of where it was found
    in the input grid. If the input grid is all zeros, it returns the original
    grid unchanged.
    
    The solution ensures that the pattern is correctly aligned from the top
    of the output grid, and handles cases where the input pattern might be
    incomplete or shorter than 3 rows. The pattern is replicated in the order:
    first row, second row, third row, repeating until the grid is filled.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the replicated pattern starting from the top.
    """
    def find_first_non_zero_row(grid):
        return next((i for i, row in enumerate(grid) if any(row)), -1)

    def extract_pattern(grid, start_row):
        pattern = []
        for i in range(start_row, min(start_row + 3, len(grid))):
            pattern.append(grid[i][:])
        while len(pattern) < 3:
            pattern.append(pattern[-1][:] if pattern else [0] * len(grid[0]))
        return pattern

    start_row = find_first_non_zero_row(input_grid.values)
    if start_row == -1:
        return input_grid  # Return original grid if it's all zeros

    pattern = extract_pattern(input_grid.values, start_row)
    output_values = [pattern[i % 3][:] for i in range(len(input_grid.values))]

    return ColoredGrid(values=output_values)
