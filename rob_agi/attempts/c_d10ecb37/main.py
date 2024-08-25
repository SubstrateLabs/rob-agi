from rob_agi.colored_grid import ColoredGrid

def solve_d10ecb37(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the d10ecb37 challenge by finding the repeating 2x2 pattern in the input grid.
    
    The function identifies the smallest 2x2 subgrid that, when repeated, can generate
    the entire input grid. This subgrid is then returned as the solution.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Check all possible 2x2 subgrids
    for r in range(2):
        for c in range(2):
            subgrid = input_grid.extract_subgrid(r, c, 2, 2)
            
            # Check if this subgrid can generate the entire grid
            if is_repeating_pattern(input_grid, subgrid):
                return subgrid
    
    # If no repeating pattern is found, return None (this should not happen for valid inputs)
    return None

def is_repeating_pattern(grid: ColoredGrid, pattern: ColoredGrid) -> bool:
    rows, cols = grid.get_dimensions()
    pattern_rows, pattern_cols = pattern.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != pattern.get_cell(r % pattern_rows, c % pattern_cols):
                return False
    
    return True
