from rob_agi.colored_grid import ColoredGrid

def solve_623ea044(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a diamond pattern centered on the colored cell,
    and then extending it diagonally to the corners of the grid.
    
    1. Find the colored cell in the input grid.
    2. Create a diamond pattern centered on that cell.
    3. Extend the pattern diagonally to the corners of the grid.
    4. Fill the main diagonal from top-left to bottom-right.
    """
    def find_colored_cell(grid):
        for i, row in enumerate(grid.values):
            for j, cell in enumerate(row):
                if cell != 0:
                    return i, j, cell
        return None, None, None

    start_row, start_col, color = find_colored_cell(input_grid)
    if color is None:
        return input_grid

    output = input_grid.deep_copy()
    height, width = output.get_dimensions()

    # Create diamond pattern
    for i in range(height):
        for j in range(width):
            if abs(i - start_row) + abs(j - start_col) <= 3:
                output.set_cell(i, j, color)

    # Extend pattern diagonally
    for i in range(height):
        output.set_cell(i, i, color)
        output.set_cell(i, width - 1 - i, color)

    return output
