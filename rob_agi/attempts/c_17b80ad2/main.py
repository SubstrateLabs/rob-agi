from rob_agi.colored_grid import ColoredGrid

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating vertical lines of color based on the following rules:
    1. Non-black dots initiate vertical lines that extend both upwards and downwards.
    2. Lines continue until they reach another non-black dot or the edge of the grid.
    3. The color changes at each non-black dot position.
    4. If there's a gap between colored segments in a column, it's filled with the color above.
    5. Gray dots in the bottom row are preserved regardless of lines above them.
    6. Columns without any non-black dots remain entirely black (0).
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    for col in range(width):
        non_black_dots = [(row, input_grid.values[row][col]) 
                          for row in range(height) 
                          if input_grid.values[row][col] != 0]
        
        if non_black_dots:
            current_color = non_black_dots[0][1]
            dot_index = 0
            
            for row in range(height):
                if dot_index < len(non_black_dots) and row == non_black_dots[dot_index][0]:
                    current_color = non_black_dots[dot_index][1]
                    dot_index += 1
                new_grid.values[row][col] = current_color

    # Handle gray dots in the bottom row
    for col in range(width):
        if input_grid.values[height-1][col] == 5:
            new_grid.values[height-1][col] = 5

    return new_grid
