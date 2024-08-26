from rob_agi.colored_grid import ColoredGrid

def solve_5ffb2104(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero elements to the right side of the grid.
    
    The transformation maintains the following properties:
    1. All non-zero elements are moved to the rightmost available columns.
    2. The vertical order of elements within each column is preserved.
    3. Elements from different columns are processed independently.
    4. The relative vertical positions of all elements are maintained.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with all non-zero elements moved to the right side.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    rightmost_column = cols - 1

    for col in range(cols):
        non_zero_elements = []
        
        for row in range(rows):
            value = input_grid.values[row][col]
            if value != 0:
                non_zero_elements.append((value, row))
        
        if non_zero_elements:
            for value, row in non_zero_elements:
                new_grid.values[row][rightmost_column] = value
            rightmost_column -= 1

    return new_grid
