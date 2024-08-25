from rob_agi.colored_grid import ColoredGrid

def solve_8597cfd7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by following these steps:
    1. Find the horizontal gray line.
    2. Scan for the leftmost vertical line that:
       - Starts from the top of the grid
       - Is continuous until the gray line
       - Extends at least one square below the gray line
    3. Return a 2x2 grid filled with the color of the found line, or black if no line is found.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Find the gray line
    gray_line_index = next(r for r in range(rows) if all(input_grid.values[r][c] == 5 for c in range(cols)))
    
    # Scan for the target vertical line
    target_color = 0
    for col in range(cols):
        top_color = input_grid.values[0][col]
        if top_color == 0:
            continue
        
        is_continuous = True
        for row in range(1, gray_line_index):
            if input_grid.values[row][col] != top_color:
                is_continuous = False
                break
        
        if is_continuous and input_grid.values[gray_line_index + 1][col] == top_color:
            target_color = top_color
            break
    
    # Create the output grid
    return ColoredGrid(values=[[target_color] * 2 for _ in range(2)])
