from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. 3x3 solid blue squares are changed to green squares.
    2. Blue "plus" shapes are changed to red "plus" shapes.
    3. Any remaining blue squares are changed to green.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_3x3_blue_square(grid, row, col):
        if row + 2 >= rows or col + 2 >= cols:
            return False
        return all(grid.values[r][c] == 1 for r in range(row, row + 3) for c in range(col, col + 3))

    def is_blue_plus(grid, row, col):
        if grid.values[row][col] != 1:
            return False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if not (0 <= row + dr < rows and 0 <= col + dc < cols) or grid.values[row + dr][col + dc] != 1:
                return False
        return True

    # First pass: Transform 3x3 blue squares to green
    for row in range(rows - 2):
        for col in range(cols - 2):
            if is_3x3_blue_square(output_grid, row, col):
                for r in range(row, row + 3):
                    for c in range(col, col + 3):
                        output_grid.values[r][c] = 3  # Green

    # Second pass: Transform blue "plus" shapes to red
    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if is_blue_plus(output_grid, row, col):
                output_grid.values[row][col] = 2  # Red
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    output_grid.values[row + dr][col + dc] = 2  # Red

    # Third pass: Change remaining blue to green
    for row in range(rows):
        for col in range(cols):
            if output_grid.values[row][col] == 1:
                output_grid.values[row][col] = 3  # Green

    return output_grid
