from rob_agi.colored_grid import ColoredGrid

def is_red_square(row, col, grid):
    return all(grid.values[r][c] == 2 for r in range(row, row + 2) for c in range(col, col + 2))

def solve_817e6c09(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing 2x2 red squares to sky blue,
    but only if they touch any edge of the grid (except the top-left corner).
    
    The function works as follows:
    1. Create a deep copy of the input grid.
    2. Iterate through each cell in the grid, considering it as the top-left corner of a 2x2 square.
    3. For each cell, if it's part of a 2x2 red square:
       - If the square is in the top-left corner, keep it red.
       - If the square touches any edge (except top-left corner), change it to sky blue.
       - If the square doesn't touch any edge, keep it red.
    4. Return the modified grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for row in range(rows - 1):
        for col in range(cols - 1):
            if is_red_square(row, col, input_grid):
                if row == 0 and col == 0:
                    continue  # Keep top-left corner red
                if row == 0 or row == rows - 2 or col == 0 or col == cols - 2:
                    for r in range(row, row + 2):
                        for c in range(col, col + 2):
                            output_grid.values[r][c] = 8

    return output_grid
