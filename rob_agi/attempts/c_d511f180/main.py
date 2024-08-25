from rob_agi.colored_grid import ColoredGrid

def solve_d511f180(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d511f180 challenge by swapping values 5 and 8 in the input grid.
    
    This function takes a ColoredGrid as input, creates a deep copy,
    and then swaps all occurrences of 5 with 8 and vice versa.
    All other values remain unchanged.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: A new grid with 5 and 8 values swapped.
    """
    def swap_5_and_8(value):
        return 8 if value == 5 else 5 if value == 8 else value
    
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    for row in range(rows):
        for col in range(cols):
            new_grid.set_cell(row, col, swap_5_and_8(new_grid.get_cell(row, col)))
    
    return new_grid
