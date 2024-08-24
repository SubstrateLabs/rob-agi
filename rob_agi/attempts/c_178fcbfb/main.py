from rob_agi.colored_grid import ColoredGrid

def solve_178fcbfb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Fills entire rows containing color 3 with 3.
    2. Fills entire rows containing color 1 with 1, overwriting any 3s if present.
    3. Fills columns containing color 2 with 2, but only in cells not already filled by 3 or 1.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the rules.
    """
    rows, cols = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find rows with 3 and 1, and columns with 2
    rows_3 = [r for r in range(rows) if 3 in input_grid.values[r]]
    rows_1 = [r for r in range(rows) if 1 in input_grid.values[r]]
    cols_2 = [c for c in range(cols) if any(input_grid.values[r][c] == 2 for r in range(rows))]
    
    # Fill rows with 3
    for r in rows_3:
        output.values[r] = [3] * cols
    
    # Fill rows with 1 (overwriting 3 if present)
    for r in rows_1:
        output.values[r] = [1] * cols
    
    # Fill columns with 2 where not already filled
    for c in cols_2:
        for r in range(rows):
            if output.values[r][c] == 0:
                output.values[r][c] = 2
    
    return output
