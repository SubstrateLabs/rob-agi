from rob_agi.colored_grid import ColoredGrid

def is_red_square(row, col, grid):
    return all(grid.values[r][c] == 2 for r in range(row, row + 2) for c in range(col, col + 2))

def should_change_to_sky_blue(row, col, rows, cols):
    if row == 0 and col == 0:
        return False  # Top-left corner remains unchanged
    return col == 0 or col == cols - 2 or row == rows - 2

def solve_817e6c09(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing 2x2 red squares to sky blue,
    but only if they touch the left, right, or bottom edges of the grid.
    The top-left corner red square, if present, remains unchanged.
    Red squares touching only the top edge or not touching any edge remain red.
    
    The function works as follows:
    1. Create a deep copy of the input grid.
    2. Iterate through each cell in the grid, considering it as the top-left corner of a 2x2 square.
    3. For each cell, if it's part of a 2x2 red square:
       - If the square is in the top-left corner, keep it red.
       - If the square touches the left, right, or bottom edge, change it to sky blue.
       - If the square touches only the top edge or doesn't touch any edge, keep it red.
    4. Return the modified grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for row in range(rows - 1):
        for col in range(cols - 1):
            if is_red_square(row, col, input_grid):
                if should_change_to_sky_blue(row, col, rows, cols):
                    for r in range(row, row + 2):
                        for c in range(col, col + 2):
                            output_grid.values[r][c] = 8

    return output_grid
