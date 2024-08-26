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
    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color != 0:
                if color not in color_groups:
                    color_groups[color] = {'left': cols, 'right': 0, 'cells': []}
                color_groups[color]['left'] = min(color_groups[color]['left'], c)
                color_groups[color]['right'] = max(color_groups[color]['right'], c)
                color_groups[color]['cells'].append((r, c))
    
    # Step 2 and 3: Process color groups and place in new grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_col = 0
    
    for color, group in sorted(color_groups.items(), key=lambda x: x[1]['left']):
        width = group['right'] - group['left'] + 1
        sorted_cells = sorted(group['cells'], key=lambda x: (x[1], -x[0]))  # Sort by column, then reverse row
        
        for i, (_, c) in enumerate(sorted_cells):
            new_c = current_col + (c - group['left'])
            new_r = rows - 1 - (i % width)
            if 0 <= new_r < rows and 0 <= new_c < cols:
                new_grid[new_r][new_c] = color
        
        current_col += width
    
    # Step 4: Final consolidation
    for c in range(cols):
        column = [new_grid[r][c] for r in range(rows)]
        non_zero = [color for color in column if color != 0]
        for r in range(rows - len(non_zero), rows):
            if r - (rows - len(non_zero)) < len(non_zero):
                new_grid[r][c] = non_zero[r - (rows - len(non_zero))]
            else:
                new_grid[r][c] = 0
    
    return ColoredGrid(values=new_grid)
