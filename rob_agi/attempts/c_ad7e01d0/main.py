from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid based on the following rules:
    1. The output grid size is the square of the input grid size.
    2. Repeat the input pattern in the top-left quadrant.
    3. Extend the pattern along the top and left edges.
    4. Fill the center with the most common border color for even-sized inputs.
    5. Ensure the bottom and right edges match the input pattern.
    """
    n = len(input_grid.values)
    output_size = n * n
    output_values = [[0 for _ in range(output_size)] for _ in range(output_size)]
    
    # Determine the most common border color
    border_colors = (
        input_grid.values[0] + input_grid.values[-1] +
        [row[0] for row in input_grid.values] + [row[-1] for row in input_grid.values]
    )
    fill_color = max(set(border_colors), key=border_colors.count)
    
    # Fill the top-left quadrant
    for i in range(n):
        for j in range(n):
            output_values[i][j] = input_grid.values[i][j]
    
    # Extend the pattern horizontally in the top n rows
    for i in range(n):
        for j in range(n, output_size):
            output_values[i][j] = output_values[i][j % n]
    
    # Extend the pattern vertically in the leftmost n columns
    for i in range(n, output_size):
        for j in range(n):
            output_values[i][j] = output_values[i % n][j]
    
    # Fill the center for even-sized inputs
    if n % 2 == 0:
        for i in range(n, output_size - n):
            for j in range(n, output_size - n):
                output_values[i][j] = fill_color
    
    # Ensure the bottom and right edges match the input pattern
    for i in range(output_size - n, output_size):
        for j in range(output_size):
            output_values[i][j] = input_grid.values[i % n][j % n]
    for i in range(output_size):
        for j in range(output_size - n, output_size):
            output_values[i][j] = input_grid.values[i % n][j % n]
    
    return ColoredGrid(values=output_values)
