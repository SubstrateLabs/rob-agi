from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. 3x3 solid blue squares remain unchanged.
    2. Blue "plus" shapes are changed to red "plus" shapes.
    3. Any other blue squares remain unchanged.
    
    The transformations are applied in the order listed above, and each rule is applied
    to the entire grid before moving to the next rule.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_3x3_blue_square(grid, row, col):
        if row + 2 >= rows or col + 2 >= cols:
            return False
        return all(grid.values[r][c] == 1 for r in range(row, row + 3) for c in range(col, col + 3))

    def is_blue_plus(grid, row, col):
        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            return False
        return (grid.values[row][col] == 1 and
                grid.values[row-1][col] == 1 and
                grid.values[row+1][col] == 1 and
                grid.values[row][col-1] == 1 and
                grid.values[row][col+1] == 1)

    # First pass: Identify 3x3 blue squares (no change needed)
    blue_squares = set()
    for row in range(rows - 2):
        for col in range(cols - 2):
            if is_3x3_blue_square(output_grid, row, col):
                for r in range(row, row + 3):
                    for c in range(col, col + 3):
                        blue_squares.add((r, c))

    # Second pass: Transform blue "plus" shapes to red, but not if part of a 3x3 blue square
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if is_blue_plus(output_grid, row, col) and (row, col) not in blue_squares:
                output_grid.values[row][col] = 2  # Red
                output_grid.values[row-1][col] = 2
                output_grid.values[row+1][col] = 2
                output_grid.values[row][col-1] = 2
                output_grid.values[row][col+1] = 2

    return output_grid
