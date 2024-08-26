from rob_agi.colored_grid import ColoredGrid

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by extending colors upwards based on the following rules:
    1. Process the grid from bottom to top.
    2. Colors extend upwards until they meet another color or reach the top of the grid.
    3. The bottom row colors are preserved and extended upwards.
    4. Columns without any colored dots remain entirely black (0).
    """
    height, width = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(width)] for _ in range(height)])

    for col in range(width):
        current_color = 0
        for row in range(height - 1, -1, -1):
            if input_grid.values[row][col] != 0:
                current_color = input_grid.values[row][col]
            new_grid.values[row][col] = current_color

    return new_grid
