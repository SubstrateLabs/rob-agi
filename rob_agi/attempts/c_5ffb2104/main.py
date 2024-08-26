from rob_agi.colored_grid import ColoredGrid

def solve_5ffb2104(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compacting all non-zero elements to the right side of the grid.
    
    The transformation maintains the following properties:
    1. Non-zero elements are moved as far right as possible.
    2. The vertical order of elements within each column is preserved.
    3. The relative horizontal order of elements in each row is maintained.
    4. Elements are compacted together without leaving empty spaces between them.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with all non-zero elements compacted to the right side.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]

    for col in range(cols):
        new_row = rows - 1  # Start placing elements from the bottom row
        for row in range(rows - 1, -1, -1):  # Iterate from bottom to top
            if input_grid.values[row][col] != 0:
                # Place the non-zero element in the bottommost available position
                new_grid[new_row][col] = input_grid.values[row][col]
                new_row -= 1  # Move the placement pointer up

    # Compact each row to the right
    for row in range(rows):
        non_zero = [val for val in new_grid[row] if val != 0]
        new_grid[row] = [0] * (cols - len(non_zero)) + non_zero

    return ColoredGrid(values=new_grid)
