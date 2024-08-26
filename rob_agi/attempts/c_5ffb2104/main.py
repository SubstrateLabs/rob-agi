from rob_agi.colored_grid import ColoredGrid

def solve_5ffb2104(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero elements to the right side of the grid.
    
    The transformation maintains the following properties:
    1. All non-zero elements are moved to the rightmost available columns.
    2. The vertical order of elements within each column is preserved.
    3. The relative horizontal order of elements is maintained.
    4. The relative vertical positions of all elements are maintained.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with all non-zero elements moved to the right side.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Collect all non-zero elements
    non_zero_elements = []
    for col in range(cols):
        for row in range(rows):
            if input_grid.values[row][col] != 0:
                non_zero_elements.append((input_grid.values[row][col], row))
    
    # Place non-zero elements from right to left
    for i, (value, row) in enumerate(reversed(non_zero_elements)):
        new_col = cols - 1 - i
        new_grid[row][new_col] = value
    
    return ColoredGrid(values=new_grid)
