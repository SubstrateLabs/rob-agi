from rob_agi.colored_grid import ColoredGrid

def solve_64a7c07e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by shifting non-black cells horizontally towards the center.
    
    The function calculates a shift for each column based on its distance from the center.
    It then creates a new grid where each non-black cell is moved to the right by the calculated shift amount.
    The vertical positions and internal structure of all shapes are preserved.
    The shift is calculated to move objects closer to the horizontal center of the grid.
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])
    
    def calculate_shift(column):
        return (width - 1 - column) // 2
    
    for r in range(height):
        for c in range(width):
            if input_grid.values[r][c] != 0:
                shift = calculate_shift(c)
                new_c = min(c + shift, width - 1)
                new_grid.values[r][new_c] = input_grid.values[r][c]
    
    return new_grid
