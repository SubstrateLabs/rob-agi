from rob_agi.colored_grid import ColoredGrid

def solve_b60334d2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 3x3 pattern around each '5' in the grid.
    The pattern is: 5 1 5, 1 0 1, 5 1 5.
    Existing non-zero values are preserved when patterns overlap.
    The center '5' of each pattern is always maintained.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid after applying the pattern to all '5's.
    """
    def apply_pattern(grid, row, col):
        pattern = [
            [5, 1, 5],
            [1, 0, 1],
            [5, 1, 5]
        ]
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_row, new_col = row + i, col + j
                if 0 <= new_row < grid.num_rows and 0 <= new_col < grid.num_cols:
                    if grid.get_cell(new_row, new_col) == 0 or (i == 0 and j == 0):
                        grid.set_cell(new_row, new_col, pattern[i+1][j+1])
    
    output = input_grid.deep_copy()
    
    for row in range(output.num_rows):
        for col in range(output.num_cols):
            if input_grid.get_cell(row, col) == 5:
                apply_pattern(output, row, col)
    
    return output
