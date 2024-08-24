from rob_agi.colored_grid import ColoredGrid

def solve_913fb3ed(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by surrounding special numbers with specific colors:
    - 3 is surrounded by 6 (magenta)
    - 2 is surrounded by 1 (blue)
    - 8 is surrounded by 4 (yellow)
    Each special number is at the center of a 3x3 area, which is filled with the corresponding color.
    The original special number remains unchanged at the center.
    """
    def apply_transformation(grid, row, col, surrounding_value):
        for i in range(max(0, row-1), min(len(grid), row+2)):
            for j in range(max(0, col-1), min(len(grid[0]), col+2)):
                if (i, j) != (row, col):  # Don't change the center
                    grid[i][j] = surrounding_value

    result = input_grid.deep_copy()
    height, width = result.get_dimensions()
    
    # Order of transformations: 3 (6's), 2 (1's), 8 (4's)
    transformations = [(3, 6), (2, 1), (8, 4)]
    
    for center_value, surrounding_value in transformations:
        for row in range(height):
            for col in range(width):
                if input_grid.get_cell(row, col) == center_value:
                    apply_transformation(result.values, row, col, surrounding_value)
    
    return result
