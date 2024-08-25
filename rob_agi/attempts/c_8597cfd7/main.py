from rob_agi.colored_grid import ColoredGrid

def solve_8597cfd7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Find the horizontal gray line.
    2. Scan from right to left for vertical lines that:
       - Extend at least one square above and below the gray line
       - Prioritize the rightmost line that meets this criteria
       - In case of multiple lines, choose the one with more squares below the gray line
    3. Return a 2x2 grid filled with the color of the chosen line, or black if no line is found.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the gray line
    gray_line_index = next(r for r in range(rows) if all(input_grid.values[r][c] == 5 for c in range(cols)))
    
    chosen_color = 0
    max_squares_below = 0

    # Scan for the target vertical line
    for col in range(cols - 1, -1, -1):  # Scan from right to left
        color_above = next((input_grid.values[r][col] for r in range(gray_line_index - 1, -1, -1) if input_grid.values[r][col] != 0), 0)
        color_below = next((input_grid.values[r][col] for r in range(gray_line_index + 1, rows) if input_grid.values[r][col] != 0), 0)
        
        if color_above != 0 and color_above == color_below:
            squares_below = sum(1 for r in range(gray_line_index + 1, rows) if input_grid.values[r][col] == color_below)
            if squares_below >= max_squares_below:
                chosen_color = color_below
                max_squares_below = squares_below
                break  # We've found the rightmost valid line

    # Create the output grid
    return ColoredGrid(values=[[chosen_color] * 2 for _ in range(2)])
