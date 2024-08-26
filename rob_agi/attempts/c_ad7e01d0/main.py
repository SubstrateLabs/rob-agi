from rob_agi.colored_grid import ColoredGrid

def solve_ad7e01d0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a larger grid based on the following rules:
    1. The output grid size is the square of the input grid size.
    2. For odd-sized inputs, repeat the pattern n times vertically and horizontally.
    3. For even-sized inputs, repeat the pattern n/2 times in the top-left quadrant,
       fill other quadrants partially, and fill the center with the most common border color.
    4. Handle special cases for bottom row and rightmost column.
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
    
    # Fill the output grid
    for i in range(output_size):
        for j in range(output_size):
            if n % 2 == 1:  # Odd-sized input
                output_values[i][j] = input_grid.values[i % n][j % n]
            else:  # Even-sized input
                if i < n * (n // 2) and j < n * (n // 2):
                    # Top-left quadrant
                    output_values[i][j] = input_grid.values[i % n][j % n]
                elif (i >= n * (n // 2) and j < n) or (i < n and j >= n * (n // 2)):
                    # Top-right and bottom-left quadrants
                    output_values[i][j] = input_grid.values[i % n][j % n]
                elif i >= output_size - n and j >= output_size - n:
                    # Bottom-right corner
                    output_values[i][j] = input_grid.values[i % n][j % n]
                else:
                    # Center and remaining areas
                    output_values[i][j] = fill_color
    
    # Special treatment for bottom row and rightmost column
    if n % 2 == 0:
        for i in range(output_size):
            output_values[i][-1] = input_grid.values[i % n][-1]
            output_values[-1][i] = input_grid.values[-1][i % n]
    
    return ColoredGrid(values=output_values)
