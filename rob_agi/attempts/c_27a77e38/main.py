from rob_agi.colored_grid import ColoredGrid

def solve_27a77e38(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the 27a77e38 challenge by modifying the input grid.
    
    The solution involves the following steps:
    1. Find the middle column of the grid.
    2. Identify all unique colors in the top row.
    3. Count the occurrences of these colors in the middle column until a gray (5) row is encountered.
    4. Choose the color with the highest count (or the leftmost in case of a tie).
    5. Place the chosen color in the bottom row of the middle column.
    
    Args:
        input_grid (ColoredGrid): The input grid to be modified.
    
    Returns:
        ColoredGrid: The modified grid with the solution applied.
    """
    # Get dimensions of the grid
    num_rows, num_cols = input_grid.get_dimensions()
    
    # Find the middle column
    middle_col = (num_cols - 1) // 2
    
    # Get unique colors from the top row
    top_row_colors = set(input_grid.values[0])
    
    # Count occurrences of top row colors in the middle column
    color_counts = {}
    max_count = 0
    chosen_color = None
    
    for row in range(num_rows):
        if input_grid.values[row][middle_col] == 5:  # Stop at gray row
            break
        color = input_grid.values[row][middle_col]
        if color in top_row_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
            if color_counts[color] > max_count:
                max_count = color_counts[color]
                chosen_color = color
    
    # If there's a tie, choose the leftmost color from the top row
    if chosen_color is None or sum(count == max_count for count in color_counts.values()) > 1:
        for color in input_grid.values[0]:
            if color in color_counts and color_counts[color] == max_count:
                chosen_color = color
                break
    
    # Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()
    
    # Modify the bottom row at the middle column
    new_grid.values[num_rows - 1][middle_col] = chosen_color
    
    return new_grid
