from rob_agi.colored_grid import ColoredGrid


def solve_c59eb873(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size and expanding each cell into a 2x2 square.
    
    This function takes an input ColoredGrid and returns a new ColoredGrid where:
    1. The dimensions are doubled (both width and height).
    2. Each cell from the input grid is expanded into a 2x2 square in the output grid.
    3. The color of each 2x2 square in the output matches the color of the corresponding input cell.
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])
    
    for input_row in range(input_rows):
        for input_col in range(input_cols):
            color = input_grid.values[input_row][input_col]
            output_row = input_row * 2
            output_col = input_col * 2
            
            output_grid.values[output_row][output_col] = color
            output_grid.values[output_row][output_col + 1] = color
            output_grid.values[output_row + 1][output_col] = color
            output_grid.values[output_row + 1][output_col + 1] = color
    
    return output_grid
