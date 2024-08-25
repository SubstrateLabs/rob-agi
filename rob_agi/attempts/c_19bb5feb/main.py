from rob_agi.colored_grid import ColoredGrid

def solve_19bb5feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 19bb5feb challenge by transforming the input grid into a 2x2 output grid.
    
    The solution involves:
    1. Scanning the input grid to find 2x2 colored squares (excluding black and sky blue).
    2. Sorting these squares based on their color value (ascending order).
    3. Creating a 2x2 output grid where:
       - The top-left cell is the color with the lowest value.
       - The top-right cell is the color with the second-lowest value (if available).
       - The bottom-left cell is the color with the highest value (if available).
       - The bottom-right cell is the middle color value if exactly three colors are found, otherwise black (0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed 2x2 output grid.
    """
    found_squares = []
    search_colors = set(range(1, 8))  # Colors 1 to 7, excluding 0 and 8

    rows, cols = input_grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            color = input_grid.values[r][c]
            if color in search_colors:
                if (color == input_grid.values[r][c+1] == 
                    input_grid.values[r+1][c] == input_grid.values[r+1][c+1]):
                    found_squares.append(color)

    found_squares.sort()  # Sort by color value

    output_values = [[0, 0], [0, 0]]
    if found_squares:
        output_values[0][0] = found_squares[0]  # Top-left: lowest color value
        if len(found_squares) > 1:
            output_values[0][1] = found_squares[1]  # Top-right: second-lowest color value
        if len(found_squares) > 2:
            output_values[1][0] = found_squares[-1]  # Bottom-left: highest color value
            if len(found_squares) == 3:
                output_values[1][1] = found_squares[1]  # Bottom-right: middle color value

    return ColoredGrid(values=output_values)
