from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_16b78196(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by reorganizing color shapes into a centralized vertical structure.
    
    The solution follows these steps:
    1. Analyze the input grid to identify colors and their characteristics.
    2. Identify background colors at the top and bottom of the grid.
    3. Create a new grid and copy background colors to their original positions.
    4. Sort non-background colors based on their average vertical position.
    5. Create a central vertical stack for each non-background color, with height proportional to its size.
    6. Fill remaining non-background area with black.
    
    Returns a new ColoredGrid with the transformed arrangement.
    """
    # Step 1: Analyze the input grid
    colors = analyze_grid(input_grid)
    
    # Step 2: Identify background colors
    bottom_bg, top_bg = identify_background_colors(input_grid)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])
    
    # Step 3: Copy background colors
    non_bg_start, non_bg_end = copy_background_colors(output, input_grid, bottom_bg, top_bg)
    
    # Step 4: Sort non-background colors
    sorted_colors = sort_colors(colors, bottom_bg, top_bg)
    
    # Step 5: Create central vertical stack
    create_vertical_stack(output, sorted_colors, colors, non_bg_start, non_bg_end)
    
    return output

def analyze_grid(grid: ColoredGrid) -> Dict[int, Dict]:
    colors = defaultdict(lambda: {'cells': [], 'avg_y': 0, 'size': 0})
    for y, row in enumerate(grid.values):
        for x, color in enumerate(row):
            if color != 0:
                colors[color]['cells'].append((x, y))
                colors[color]['size'] += 1
    
    for color, data in colors.items():
        data['avg_y'] = sum(y for _, y in data['cells']) / len(data['cells'])
    
    return colors

def identify_background_colors(grid: ColoredGrid) -> Tuple[int, int]:
    bottom_row = grid.values[-1]
    top_row = grid.values[0]
    bottom_bg = max(set(bottom_row), key=bottom_row.count) if any(bottom_row) else 0
    top_bg = max(set(top_row), key=top_row.count) if any(top_row) else 0
    return bottom_bg, top_bg

def copy_background_colors(output: ColoredGrid, input_grid: ColoredGrid, bottom_bg: int, top_bg: int) -> Tuple[int, int]:
    non_bg_start, non_bg_end = 0, 30
    
    # Copy bottom background
    if bottom_bg:
        for y in range(len(input_grid.values) - 1, -1, -1):
            if input_grid.values[y].count(bottom_bg) / len(input_grid.values[y]) < 0.9:
                non_bg_end = y + 1
                break
            output.values[y] = input_grid.values[y].copy()
    
    # Copy top background
    if top_bg:
        for y in range(len(input_grid.values)):
            if input_grid.values[y].count(top_bg) / len(input_grid.values[y]) < 0.9:
                non_bg_start = y
                break
            output.values[y] = input_grid.values[y].copy()
    
    return non_bg_start, non_bg_end

def sort_colors(colors: Dict[int, Dict], bottom_bg: int, top_bg: int) -> List[int]:
    return sorted([c for c in colors if c not in {bottom_bg, top_bg, 0}], key=lambda c: colors[c]['avg_y'])

def create_vertical_stack(output: ColoredGrid, sorted_colors: List[int], colors: Dict[int, Dict], non_bg_start: int, non_bg_end: int):
    available_height = non_bg_end - non_bg_start
    total_size = sum(colors[c]['size'] for c in sorted_colors)
    
    start_y = non_bg_start
    for color in sorted_colors:
        color_height = max(1, round((colors[color]['size'] / total_size) * available_height))
        for y in range(start_y, min(start_y + color_height, non_bg_end)):
            for x in range(13, 16):
                output.values[y][x] = color
        start_y += color_height
