from rob_agi.colored_grid import ColoredGrid

def solve_50a16a69(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the core checkerboard pattern and extending it to cover the entire grid.
    
    The function performs the following steps:
    1. Identifies the two colors of the checkerboard pattern from the top-left 2x2 subgrid.
    2. Creates a new grid by extending the checkerboard pattern across the entire area, including original borders.
    
    This approach works for checkerboard patterns, handling various grid sizes and extending the pattern
    to areas that were originally borders or uniform regions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with the extended checkerboard pattern.
    """
    height, width = input_grid.get_dimensions()
    
    # Identify the two colors of the checkerboard pattern
    color1 = input_grid.values[0][0]
    color2 = input_grid.values[0][1] if input_grid.values[0][1] != color1 else input_grid.values[1][0]
    
    output_values = []
    for i in range(height):
        row = []
        for j in range(width):
            color = color1 if (i + j) % 2 == 0 else color2
            row.append(color)
        output_values.append(row)
    
    return ColoredGrid(values=output_values)
