
from rob_agi.colored_grid import ColoredGrid

def solve_c59eb873(input_grid: ColoredGrid) -> ColoredGrid:
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    
    output_values = [[0 for _ in range(output_cols)] for _ in range(output_rows)]
    
    for i in range(input_rows):
        for j in range(input_cols):
            color = input_grid.values[i][j]
            output_values[2*i][2*j] = color
            output_values[2*i][2*j+1] = color
            output_values[2*i+1][2*j] = color
            output_values[2*i+1][2*j+1] = color
    
    return ColoredGrid(values=output_values)
