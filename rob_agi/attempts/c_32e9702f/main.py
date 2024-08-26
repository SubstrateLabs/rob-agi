from rob_agi.colored_grid import ColoredGrid

def solve_32e9702f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Replace all black (0) cells with gray (5) cells.
    2. Preserve non-black shapes, except for the rightmost cell which becomes gray.
    3. Protect cells directly below non-black cells by keeping their original value.
    4. Turn cells to the left and right of non-black shapes to gray if they were originally black.
    5. If there's a 2x2 yellow (4) square in the top-left corner, expand it diagonally by one cell.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid.
    """
    # Step 1: Create a deep copy
    result_grid = input_grid.deep_copy()
    
    # Step 2: Create a new grid filled with gray
    rows, cols = result_grid.get_dimensions()
    gray_grid = ColoredGrid(values=[[5 for _ in range(cols)] for _ in range(rows)])
    
    # Step 3: Main transformation
    for r in range(rows):
        start_col = None
        for c in range(cols):
            if result_grid.values[r][c] != 0:
                if start_col is None:
                    start_col = c
                gray_grid.values[r][c] = result_grid.values[r][c]
                if r + 1 < rows:
                    gray_grid.values[r+1][c] = result_grid.values[r+1][c]
            elif start_col is not None:
                if c > start_col:
                    gray_grid.values[r][c-1] = 5
                start_col = None
        if start_col is not None and cols > start_col:
            gray_grid.values[r][cols-1] = 5
    
    # Step 4: Handle yellow expansion
    handle_yellow_expansion(gray_grid)
    
    return gray_grid

def handle_yellow_expansion(grid: ColoredGrid):
    if (len(grid.values) > 2 and len(grid.values[0]) > 2 and
        grid.values[0][0] == 4 and grid.values[0][1] == 4 and
        grid.values[1][0] == 4 and grid.values[1][1] == 4):
        grid.values[2][2] = 4
