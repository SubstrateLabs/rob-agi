from rob_agi.colored_grid import ColoredGrid

def solve_5ffb2104(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compacting all non-zero elements to the right side of the grid.
    
    The transformation maintains the following properties:
    1. Non-zero elements are moved as far right as possible.
    2. The vertical order of elements within each column is preserved.
    3. The relative horizontal order of elements across columns is maintained.
    4. Elements are compacted together without leaving empty spaces between them.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A new grid with all non-zero elements compacted to the right side.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    right_col = cols - 1

    for col in range(cols):
        non_zero_elements = []
        for row in range(rows):
            if input_grid.values[row][col] != 0:
                non_zero_elements.append((input_grid.values[row][col], row))
        
        if non_zero_elements:
            non_zero_elements.sort(key=lambda x: x[1])  # Sort by original row
            for value, original_row in non_zero_elements:
                new_grid[original_row][right_col] = value
            right_col -= 1

    return ColoredGrid(values=new_grid)
