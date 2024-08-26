from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. 3x3 solid blue squares remain unchanged.
    2. Blue "plus" shapes are changed to red or green (whichever color exists in the grid).
    3. 2x2 blue squares are changed to red or green (whichever color exists in the grid).
    4. Any other blue squares remain unchanged.
    
    The transformations are applied in the order listed above, and each rule is applied
    to the entire grid before moving to the next rule.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Determine target color (red if it exists, otherwise green)
    target_color = 2 if any(2 in row for row in output_grid.values) else 3

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

    def is_2x2_blue_square(grid, row, col):
        if row + 1 >= rows or col + 1 >= cols:
            return False
        return all(grid.values[r][c] == 1 for r in range(row, row + 2) for c in range(col, col + 2))

    # First pass: Identify 3x3 blue squares (no change needed)
    blue_squares = set()
    for row in range(rows - 2):
        for col in range(cols - 2):
            if is_3x3_blue_square(output_grid, row, col):
                for r in range(row, row + 3):
                    for c in range(col, col + 3):
                        blue_squares.add((r, c))

    # Second pass: Transform blue shapes
    for row in range(rows):
        for col in range(cols):
            if (row, col) not in blue_squares and output_grid.values[row][col] == 1:
                if is_blue_plus(output_grid, row, col):
                    for r, c in [(row, col), (row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                        output_grid.values[r][c] = target_color
                elif is_2x2_blue_square(output_grid, row, col):
                    for r in range(row, min(row + 2, rows)):
                        for c in range(col, min(col + 2, cols)):
                            output_grid.values[r][c] = target_color

    return output_grid
