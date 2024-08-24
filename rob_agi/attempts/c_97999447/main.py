from rob_agi.colored_grid import ColoredGrid

def solve_97999447(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by extending horizontal patterns from non-zero starting points.
    
    For each non-zero number in the input grid:
    1. Use it as a starting point for a horizontal pattern.
    2. Extend the pattern to the right edge of the grid.
    3. Alternate between the original number and 5 in the pattern.
    
    Return the transformed grid.
    """
    height, width = input_grid.get_dimensions()
    output = input_grid.deep_copy()
    
    for row in range(height):
        for col in range(width):
            value = input_grid.get_cell(row, col)
            if value != 0:
                for i in range(col, width):
                    if (i - col) % 2 == 0:
                        output.set_cell(row, i, value)
                    else:
                        output.set_cell(row, i, 5)
    
    return output
