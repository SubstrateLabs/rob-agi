from rob_agi.colored_grid import ColoredGrid

def solve_1e0a9b12(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 1e0a9b12 challenge by applying gravity to each column independently.
    
    This function processes each column from left to right, moving non-zero elements
    to the bottom of the column while maintaining their relative order. The result
    is a grid where all non-zero elements have "fallen" to the lowest possible position
    in their respective columns.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid after applying the gravity effect.
    """
    rows, cols = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    for col in range(cols):
        bottom_row = rows - 1
        for row in range(rows - 1, -1, -1):
            cell_value = input_grid.get_cell(row, col)
            if cell_value != 0:
                output.set_cell(bottom_row, col, cell_value)
                bottom_row -= 1
    
    return output
