from rob_agi.colored_grid import ColoredGrid

def solve_8597cfd7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Find the horizontal gray line.
    2. Scan for the rightmost vertical line that:
       - Is continuous from its start until the gray line
       - Extends at least one square below the gray line
       - Is the longest such line (in case of ties, choose the rightmost)
    3. Return a 2x2 grid filled with the color of the found line, or black if no line is found.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the gray line
    gray_line_index = next(r for r in range(rows) if all(input_grid.values[r][c] == 5 for c in range(cols)))
    
    max_length_above = 0
    chosen_color = 0

    # Scan for the target vertical line
    for col in range(cols - 1, -1, -1):  # Scan from right to left
        current_color = 0
        current_length = 0
        for row in range(gray_line_index):
            if input_grid.values[row][col] == current_color:
                current_length += 1
            else:
                current_color = input_grid.values[row][col]
                current_length = 1
        
        # Check if the line extends below the gray line
        if input_grid.values[gray_line_index + 1][col] == current_color:
            if current_length >= max_length_above and current_color != 0:
                max_length_above = current_length
                chosen_color = current_color
    
    # Create the output grid
    return ColoredGrid(values=[[chosen_color] * 2 for _ in range(2)])
