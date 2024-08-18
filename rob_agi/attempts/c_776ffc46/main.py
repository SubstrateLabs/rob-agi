
from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    
    def change_blue_to_red(grid):
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 1:  # If the cell is blue
                    grid.values[r][c] = 2  # Change it to red
    
    # Apply the transformation
    change_blue_to_red(output_grid)
    
    return output_grid
