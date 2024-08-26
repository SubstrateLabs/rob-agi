from rob_agi.colored_grid import ColoredGrid

def solve_9ddd00f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by creating horizontal symmetry for non-zero values.
    
    The function processes the grid as follows:
    1. Determines the center of the grid.
    2. For each row, creates symmetry around the center:
       - Preserves the center column value for odd-width grids.
       - Mirrors non-zero values from left to right and right to left.
       - Preserves zero values in their original positions.
    
    This creates a pattern where horizontal symmetry is achieved for non-zero values,
    while maintaining the original structure and preserving zero values.
    """
    height, width = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    center = width // 2

    for row in range(height):
        if width % 2 != 0:
            output_grid.values[row][center] = input_grid.values[row][center]
        
        for col in range(center + (width % 2)):
            left_value = input_grid.values[row][col]
            right_value = input_grid.values[row][-(col+1)]
            
            if left_value != 0:
                output_grid.values[row][-(col+1)] = left_value
            if right_value != 0:
                output_grid.values[row][col] = right_value
            if left_value == 0:
                output_grid.values[row][col] = 0
            if right_value == 0:
                output_grid.values[row][-(col+1)] = 0

    return output_grid
