from rob_agi.colored_grid import ColoredGrid

def solve_d017b73f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by compressing it both vertically and horizontally while preserving color groups.
    
    The function processes each column from left to right, compressing non-black cells vertically
    and skipping entirely black columns. This maintains the vertical order within each column and
    the left-to-right order of color groups, while producing the narrowest possible output grid
    with an upward shift of color groups.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = []
    
    for col in range(cols):
        column = [input_grid.values[row][col] for row in range(rows)]
        if any(cell != 0 for cell in column):
            compressed_column = [cell for cell in column if cell != 0]
            compressed_column += [0] * (rows - len(compressed_column))
            new_grid.append(compressed_column)
    
    # Transpose the grid back to row-major order
    new_grid = list(map(list, zip(*new_grid)))
    
    # Trim trailing black columns
    while new_grid and all(row[-1] == 0 for row in new_grid):
        for row in new_grid:
            row.pop()
    
    return ColoredGrid(values=new_grid)
