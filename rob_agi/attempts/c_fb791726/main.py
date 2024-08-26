from rob_agi.colored_grid import ColoredGrid

def solve_fb791726(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by:
    1. Doubling its size in both dimensions
    2. Rearranging the quadrants: top-left stays, top-right moves to bottom-left,
       bottom-left moves to top-right, bottom-right stays
    3. Adding green (3) separator rows and columns between the original rows and columns
    4. Filling the rest with black (0)
    """
    input_rows, input_cols = input_grid.get_dimensions()
    output_rows, output_cols = input_rows * 2, input_cols * 2
    output_grid = ColoredGrid(values=[[0 for _ in range(output_cols)] for _ in range(output_rows)])

    def map_coordinates(x, y):
        quadrant_rows = (input_rows + 1) // 2
        quadrant_cols = (input_cols + 1) // 2
        
        if x < quadrant_rows:
            if y < quadrant_cols:
                new_x, new_y = x, y
            else:
                new_x, new_y = x + quadrant_rows + 1, y - quadrant_cols
        else:
            if y < quadrant_cols:
                new_x, new_y = x - quadrant_rows, y + quadrant_cols + 1
            else:
                new_x, new_y = x + 1, y + 1
        
        new_x += new_x // quadrant_rows
        new_y += new_y // quadrant_cols
        
        return new_x, new_y

    # Copy non-black squares from input to output
    for x in range(input_rows):
        for y in range(input_cols):
            if input_grid.values[x][y] != 0:
                new_x, new_y = map_coordinates(x, y)
                output_grid.values[new_x][new_y] = input_grid.values[x][y]

    # Add green separators
    for i in range(1, output_rows, 2):
        output_grid.values[i] = [3] * output_cols
    for j in range(1, output_cols, 2):
        for i in range(output_rows):
            output_grid.values[i][j] = 3

    return output_grid
