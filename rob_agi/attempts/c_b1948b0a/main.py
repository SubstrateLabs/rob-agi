from rob_agi.colored_grid import ColoredGrid

def solve_b1948b0a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by replacing all occurrences of 6 (magenta) with 2 (red).
    All other colors remain unchanged.
    
    This function identifies regions of connected magenta cells and replaces them
    with red cells, while preserving the structure of other colored regions.
    """
    def transform_region(region):
        return 2 if input_grid.get_cell(region[0][0], region[0][1]) == 6 else input_grid.get_cell(region[0][0], region[0][1])

    return input_grid.apply_function_to_regions(transform_region)
