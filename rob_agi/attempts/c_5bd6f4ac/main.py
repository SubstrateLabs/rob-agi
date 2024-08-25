from rob_agi.colored_grid import ColoredGrid

def solve_5bd6f4ac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts a 3x3 subgrid from the top-right corner of the input grid.
    
    The function takes the first 3 rows and the last 3 columns of the input grid
    to form the output grid. This pattern works for all given examples and the test case.
    """
    rows, cols = input_grid.get_dimensions()
    return input_grid.extract_subgrid(top=0, left=cols-3, height=3, width=3)
