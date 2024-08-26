from rob_agi.colored_grid import ColoredGrid

def is_corner(row, col, grid):
    rows, cols = grid.get_dimensions()
    return (row == 0 or row == rows - 1) and (col == 0 or col == cols - 1)

def is_edge(row, col, grid):
    rows, cols = grid.get_dimensions()
    return row == 0 or row == rows - 1 or col == 0 or col == cols - 1

def is_red_rectangle(row, col, grid):
    rows, cols = grid.get_dimensions()
    if row + 1 >= rows or col + 1 >= cols:
        return False
    return all(grid.values[r][c] == 2 for r in range(row, row + 2) for c in range(col, col + 2))

def solve_817e6c09(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by changing 2x2 red rectangles to sky blue,
    except for those in the top row or rightmost column.
    
    The function works as follows:
    1. Create a deep copy of the input grid.
    2. Iterate through each cell in the grid, considering it as the top-left corner of a 2x2 rectangle.
    3. For each cell, if it's part of a 2x2 red rectangle:
       - If the rectangle is in the top row, keep it red.
       - If the rectangle is in the rightmost column, keep it red.
       - Otherwise, change it to sky blue.
    4. Return the modified grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for row in range(rows - 1):  # -1 because we're checking 2x2 rectangles
        for col in range(cols - 1):
            if is_red_rectangle(row, col, input_grid):
                if row == 0 or col == cols - 2:
                    continue  # Keep red if in top row or rightmost column
                else:
                    # Change to sky blue
                    for r in range(row, row + 2):
                        for c in range(col, col + 2):
                            output_grid.values[r][c] = 8

    return output_grid
