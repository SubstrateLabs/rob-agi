from rob_agi.colored_grid import ColoredGrid

def solve_19bb5feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 19bb5feb challenge by transforming the input grid into a 2x2 output grid.
    
    The solution involves:
    1. Scanning the input grid to find 2x2 colored squares (excluding black and sky blue).
    2. Sorting these squares based on their vertical position (top to bottom).
    3. Creating a 2x2 output grid where:
       - The top-left cell is the color of the topmost found square.
       - The top-right cell is the color of the second found square.
       - The bottom-left cell is the color of the bottommost found square.
       - The bottom-right cell remains black (0).

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed 2x2 output grid.

    Raises:
    ValueError: If exactly three colored squares are not found in the input grid.
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
                    found_squares.append((color, r))

    found_squares.sort(key=lambda x: x[1])  # Sort by row number
    if len(found_squares) != 3:
        raise ValueError("Invalid input: expected exactly 3 colored squares")

    output_values = [[0, 0], [0, 0]]
    output_values[0][0] = found_squares[0][0]  # Top-left
    output_values[0][1] = found_squares[1][0]  # Top-right
    output_values[1][0] = found_squares[2][0]  # Bottom-left

    return ColoredGrid(values=output_values)
