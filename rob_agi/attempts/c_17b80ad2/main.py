from rob_agi.colored_grid import ColoredGrid

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colors within each column based on the following rules:
    1. Process each column independently.
    2. Identify all non-zero colors in the column, preserving their vertical order.
    3. Fill the column from top to bottom:
       - Start with the first color and fill downwards until reaching the next color's position or the bottom.
       - Continue with subsequent colors, filling their respective sections.
       - If all colors are used before reaching the bottom, continue with the last color.
    4. Columns without any colored dots remain entirely black (0).
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    for col in range(width):
        # Identify colors in the column
        colors = [(row, input_grid.values[row][col]) 
                  for row in range(height) 
                  if input_grid.values[row][col] != 0]
        
        if not colors:
            continue  # Skip columns with no colors
        
        # Fill the column
        color_index = 0
        for row in range(height):
            if color_index < len(colors) - 1 and row >= colors[color_index + 1][0]:
                color_index += 1
            new_grid.values[row][col] = colors[color_index][1]

    return new_grid
