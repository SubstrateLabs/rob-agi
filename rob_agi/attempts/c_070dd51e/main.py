from rob_agi.colored_grid import ColoredGrid

def solve_070dd51e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting the outermost dots of each color with rectangles.
    
    This function finds the outermost positions of each color in the grid and
    draws rectangles to connect them. Vertical lines take precedence over horizontal lines.
    The solution works for all colors (1-9) and grid sizes up to 30x30.
    """
    color_positions = {color: {'min_row': float('inf'), 'max_row': -1, 
                               'min_col': float('inf'), 'max_col': -1} 
                       for color in range(1, 10)}
    
    # Find outermost positions for each color
    for row in range(len(input_grid.values)):
        for col in range(len(input_grid.values[0])):
            color = input_grid.values[row][col]
            if color != 0:
                color_positions[color]['min_row'] = min(color_positions[color]['min_row'], row)
                color_positions[color]['max_row'] = max(color_positions[color]['max_row'], row)
                color_positions[color]['min_col'] = min(color_positions[color]['min_col'], col)
                color_positions[color]['max_col'] = max(color_positions[color]['max_col'], col)
    
    # Create a deep copy of the input grid
    new_grid = input_grid.deep_copy()
    
    # Draw rectangles for each color
    for color, positions in color_positions.items():
        if positions['min_row'] != float('inf'):
            min_row, max_row = positions['min_row'], positions['max_row']
            min_col, max_col = positions['min_col'], positions['max_col']
            
            # Draw vertical lines (take precedence)
            for row in range(min_row, max_row + 1):
                if new_grid.values[row][min_col] == 0 or new_grid.values[row][min_col] > color:
                    new_grid.values[row][min_col] = color
                if min_col != max_col:
                    if new_grid.values[row][max_col] == 0 or new_grid.values[row][max_col] > color:
                        new_grid.values[row][max_col] = color
            
            # Draw horizontal lines
            for col in range(min_col, max_col + 1):
                if new_grid.values[min_row][col] == 0:
                    new_grid.values[min_row][col] = color
                if min_row != max_row:
                    if new_grid.values[max_row][col] == 0:
                        new_grid.values[max_row][col] = color
    
    return new_grid
