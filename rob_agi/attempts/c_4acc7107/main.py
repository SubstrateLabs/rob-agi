from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, deque

def solve_4acc7107(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Analyzes the input grid and groups cells by color
    2. Processes each color group:
       - Maintains the original width and structure of the color group
       - Vertically flips the group while consolidating disconnected parts
    3. Places processed color groups in a new grid:
       - Preserves the left-to-right order of colors
       - Ensures all color groups touch the bottom of the grid
       - Creates a "skyline" effect by removing vertical gaps
    4. Consolidates any remaining disconnected parts of the same color
    
    The transformation maintains the width of each color group, preserves left-to-right order,
    and ensures no empty rows between colored cells of the same group.
    """
    rows, cols = input_grid.get_dimensions()
    color_groups = {}
    
    # Step 1: Analyze the input grid
    for c in range(cols):
        for r in range(rows):
            color = input_grid.get_cell(r, c)
            if color != 0:
                if color not in color_groups:
                    color_groups[color] = {'left': c, 'right': c, 'rows': []}
                color_groups[color]['right'] = max(color_groups[color]['right'], c)
                color_groups[color]['rows'].append(r)
    
    # Step 2 and 3: Process color groups and place in new grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_col = 0
    
    for color, group in sorted(color_groups.items(), key=lambda x: x[1]['left']):
        width = group['right'] - group['left'] + 1
        height = len(set(group['rows']))
        bottom_row = rows - 1
        
        # Sort rows in descending order for vertical flip
        sorted_rows = sorted(set(group['rows']), reverse=True)
        
        for i, r in enumerate(sorted_rows):
            for c in range(group['left'], group['right'] + 1):
                if input_grid.get_cell(r, c) == color:
                    new_c = current_col + (c - group['left'])
                    new_r = bottom_row - (i % height)
                    new_grid[new_r][new_c] = color
        
        current_col += width
    
    # Step 4: Final consolidation
    for c in range(cols):
        column = [new_grid[r][c] for r in range(rows)]
        non_zero = [color for color in column if color != 0]
        for r in range(rows - len(non_zero), rows):
            new_grid[r][c] = non_zero[r - (rows - len(non_zero))] if r - (rows - len(non_zero)) < len(non_zero) else 0
    
    return ColoredGrid(values=new_grid)
