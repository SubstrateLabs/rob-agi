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

    # Step 2: Process columns from left to right
    for col in range(cols):
        new_row = rows - 1  # Start from the bottom of the new column
        for row in range(rows - 1, -1, -1):  # Iterate from bottom to top
            if input_grid.values[row][col] != 0:
                new_grid[new_row][col] = input_grid.values[row][col]
                new_row -= 1

    # Step 3: Compact rows to the right
    for row in range(rows):
        non_zero = [val for val in new_grid[row] if val != 0]
        new_grid[row] = [0] * (cols - len(non_zero)) + non_zero

    # Step 4: Create and return the final ColoredGrid
    return ColoredGrid(values=new_grid)
