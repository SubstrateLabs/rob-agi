from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f9a67cb5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a red structure that connects all blue squares.
    
    1. Determine the main axis (horizontal or vertical) based on blue line count.
    2. Create a main red line along the side with more blue squares.
    3. Add perpendicular extensions to connect other blue squares.
    4. Connect any isolated blue squares to the red structure.
    5. Ensure the initial red square (if present) is connected to the structure.
    
    Returns a new grid with the red structure added while preserving all blue squares.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    blue_regions = input_grid.find_connected_regions(8)
    
    # Count horizontal and vertical blue lines
    horizontal_lines = sum(1 for region in blue_regions if len(set(r for r, _ in region)) == 1)
    vertical_lines = sum(1 for region in blue_regions if len(set(c for _, c in region)) == 1)
    
    # Determine main axis and side for red line
    if horizontal_lines >= vertical_lines:
        main_axis = 'horizontal'
        left_blue = sum(1 for r, c in [(r, c) for region in blue_regions for r, c in region] if c < cols // 2)
        right_blue = sum(1 for r, c in [(r, c) for region in blue_regions for r, c in region] if c >= cols // 2)
        main_side = 'right' if right_blue > left_blue else 'left'
    else:
        main_axis = 'vertical'
        top_blue = sum(1 for r, c in [(r, c) for region in blue_regions for r, c in region] if r < rows // 2)
        bottom_blue = sum(1 for r, c in [(r, c) for region in blue_regions for r, c in region] if r >= rows // 2)
        main_side = 'bottom' if bottom_blue > top_blue else 'top'
    
    # Create main red line
    if main_axis == 'horizontal':
        col = cols - 1 if main_side == 'right' else 0
        for r in range(rows):
            if output_grid.values[r][col] != 8:
                output_grid.values[r][col] = 2
    else:
        row = rows - 1 if main_side == 'bottom' else 0
        for c in range(cols):
            if output_grid.values[row][c] != 8:
                output_grid.values[row][c] = 2
    
    # Add perpendicular extensions
    if main_axis == 'horizontal':
        for c in range(cols):
            if output_grid.values[0][c] != 8:
                output_grid.values[0][c] = 2
            if output_grid.values[rows-1][c] != 8:
                output_grid.values[rows-1][c] = 2
    else:
        for r in range(rows):
            if output_grid.values[r][0] != 8:
                output_grid.values[r][0] = 2
            if output_grid.values[r][cols-1] != 8:
                output_grid.values[r][cols-1] = 2
    
    # Connect isolated blue squares
    for region in blue_regions:
        if not any(output_grid.values[r][c] == 2 for r, c in region):
            r, c = region[0]
            if main_axis == 'horizontal':
                for col in range(min(c, cols-1-c), max(c, cols-1-c)+1):
                    if output_grid.values[r][col] != 8:
                        output_grid.values[r][col] = 2
            else:
                for row in range(min(r, rows-1-r), max(r, rows-1-r)+1):
                    if output_grid.values[row][c] != 8:
                        output_grid.values[row][c] = 2
    
    # Connect initial red square if present
    initial_red = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 2), None)
    if initial_red:
        r, c = initial_red
        if main_axis == 'horizontal':
            for col in range(min(c, cols-1), cols):
                if output_grid.values[r][col] != 8:
                    output_grid.values[r][col] = 2
        else:
            for row in range(min(r, rows-1), rows):
                if output_grid.values[row][c] != 8:
                    output_grid.values[row][c] = 2
    
    return output_grid
