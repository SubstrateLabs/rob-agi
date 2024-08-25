from rob_agi.colored_grid import ColoredGrid

def solve_070dd51e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting pairs of same-colored dots with lines.
    
    This function finds the outermost positions of each color in the grid and
    draws lines to connect them, both horizontally and vertically. The lines
    fill in all spaces between the original dots of the same color.
    """
    color_positions = {color: {'min_row': float('inf'), 'max_row': -1, 
                               'min_col': float('inf'), 'max_col': -1} 
                       for color in range(1, 10)}
    
    # First pass: find min and max positions for each color
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
    
    # Second pass: draw lines for each color
    for color, positions in color_positions.items():
        if positions['min_row'] != float('inf'):
            min_row, max_row = positions['min_row'], positions['max_row']
            min_col, max_col = positions['min_col'], positions['max_col']
            
            # Draw vertical line
            if min_row != max_row:
                for row in range(min_row, max_row + 1):
                    new_grid.values[row][min_col] = color
            
            # Draw horizontal line
            if min_col != max_col:
                for col in range(min_col, max_col + 1):
                    new_grid.values[min_row][col] = color
    
    return new_grid
