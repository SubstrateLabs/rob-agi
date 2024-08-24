from rob_agi.colored_grid import ColoredGrid

def solve_a85d4709(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid based on the position of the number 5 in each row.
    
    The transformation rules are:
    - If 5 is in the first column, the output row is [2, 2, 2]
    - If 5 is in the second column, the output row is [4, 4, 4]
    - If 5 is in the third column, the output row is [3, 3, 3]
    
    Args:
    input_grid (ColoredGrid): A 3x3 grid where each row contains a single 5 and the rest are 0s.
    
    Returns:
    ColoredGrid: A 3x3 grid with transformed values based on the position of 5 in each input row.
    """
    def transform_row(row):
        if 5 in row:
            index = row.index(5)
            return [2, 4, 3][index] * 3
        return row  # This case should not occur based on the given examples

    transformed_values = [transform_row(row) for row in input_grid.values]
    return ColoredGrid(values=transformed_values)
