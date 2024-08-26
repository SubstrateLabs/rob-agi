from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict, deque

def solve_4acc7107(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Analyzes the input grid and groups cells by color
    2. Processes each color group:
       - Vertically flips all cells within the group
       - Maintains the original width of the color group
       - Preserves the left-to-right order of colors
    3. Places processed color groups in a new grid:
       - Aligns the bottom of each group with the bottom of the grid
       - Stacks disconnected parts vertically within the group's width constraints
    4. Adjusts vertical positions to create a "skyline" effect
    5. Consolidates disconnected parts of the same color
    
    The transformation maintains the width of each color group, preserves left-to-right order,
    and ensures no empty rows between colored cells of the same group.
    """
    rows, cols = input_grid.get_dimensions()
    color_groups = defaultdict(list)
    
    # Step 1: Analyze the input grid
    for r in range(rows):
        for c in range(cols):
            color = input_grid.get_cell(r, c)
            if color != 0:
                color_groups[color].append((r, c))
    
    # Helper functions
    def vertical_flip(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(rows - 1 - r, c) for r, c in coords]
    
    def get_group_dimensions(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        min_r = min(r for r, _ in coords)
        max_r = max(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        max_c = max(c for _, c in coords)
        return min_r, max_r, min_c, max_c
    
    # Step 2: Process each color group
    processed_groups = []
    for color, coords in sorted(color_groups.items(), key=lambda x: min(c for _, c in x[1])):
        flipped_coords = vertical_flip(coords)
        _, _, min_c, max_c = get_group_dimensions(coords)
        processed_groups.append((color, flipped_coords, min_c, max_c))
    
    # Step 3: Place processed color groups in a new grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_col = 0
    for color, coords, min_c, max_c in processed_groups:
        width = max_c - min_c + 1
        group_height = max(r for r, _ in coords) - min(r for r, _ in coords) + 1
        bottom_row = rows - 1
        
        # Place the color group
        for r, c in coords:
            new_c = current_col + (c - min_c)
            new_r = bottom_row - (group_height - 1 - (r - min(r for r, _ in coords)))
            if 0 <= new_r < rows and 0 <= new_c < cols:
                new_grid[new_r][new_c] = color
        
        current_col += width
    
    # Step 4: Adjust vertical positions
    for c in range(cols):
        non_zero_cells = [(r, new_grid[r][c]) for r in range(rows) if new_grid[r][c] != 0]
        for i, (r, color) in enumerate(reversed(non_zero_cells)):
            new_grid[rows - 1 - i][c] = color
            if r != rows - 1 - i:
                new_grid[r][c] = 0
    
    return ColoredGrid(values=new_grid)
