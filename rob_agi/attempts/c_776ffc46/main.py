
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 1:  # If the color is blue (1)
                output_grid.values[r][c] = 2   # Change it to red (2)
    
    return output_grid
