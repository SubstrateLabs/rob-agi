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

    for row in range(rows):
        new_col = cols - 1  # Start placing elements from the rightmost column
        for col in range(cols - 1, -1, -1):  # Iterate from right to left
            if input_grid.values[row][col] != 0:
                # Place the non-zero element in the rightmost available position
                new_grid[row][new_col] = input_grid.values[row][col]
                new_col -= 1  # Move the placement pointer left

    return ColoredGrid(values=new_grid)
