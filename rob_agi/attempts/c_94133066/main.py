from rob_agi.colored_grid import ColoredGrid

def solve_94133066(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extracting the blue rectangle containing all non-black squares.
    
    1. Finds the boundaries of the blue rectangle in the input grid.
    2. Creates a new grid with the dimensions of the blue rectangle.
    3. Copies the contents of the blue rectangle from the input grid to the new grid.
    
    Returns a new ColoredGrid object representing the extracted blue rectangle.
    """
    rows, cols = input_grid.get_dimensions()
    min_row, max_row = rows, 0
    min_col, max_col = cols, 0

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:  # Blue cell
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    new_grid = ColoredGrid(values=[[1 for _ in range(width)] for _ in range(height)])

    for r in range(height):
        for c in range(width):
            new_grid.values[r][c] = input_grid.values[min_row + r][min_col + c]

    return new_grid
