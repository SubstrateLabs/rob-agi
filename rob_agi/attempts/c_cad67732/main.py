from rob_agi.colored_grid import ColoredGrid

def solve_cad67732(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by doubling its size and extending the pattern.
    
    The function:
    1. Creates a new grid double the size of the input.
    2. Treats the input grid as an infinitely repeating pattern.
    3. Fills the new grid by mapping each cell to the corresponding cell in the infinitely repeating pattern.
    
    This approach works for all cases by extending the pattern of the input grid,
    maintaining the relative positions and spacings of all elements.
    """
    input_height, input_width = input_grid.get_dimensions()
    new_height = input_height * 2
    new_width = input_width * 2
    
    new_grid = ColoredGrid(values=[[0 for _ in range(new_width)] for _ in range(new_height)])
    
    for row in range(new_height):
        for col in range(new_width):
            pattern_row = row % input_height
            pattern_col = col % input_width
            new_grid.values[row][col] = input_grid.values[pattern_row][pattern_col]
    
    return new_grid
