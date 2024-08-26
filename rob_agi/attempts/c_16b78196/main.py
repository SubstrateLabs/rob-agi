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
    5. Create a central vertical stack of 3x3 squares for each non-background color.
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
    copy_background_colors(output, input_grid, bottom_bg, top_bg)
    
    # Step 4: Sort non-background colors
    sorted_colors = sort_colors(colors, bottom_bg, top_bg)
    
    # Step 5: Create central vertical stack
    create_vertical_stack(output, sorted_colors, bottom_bg, top_bg)
    
    return output

def analyze_grid(grid: ColoredGrid) -> Dict[int, Dict]:
    colors = defaultdict(lambda: {'cells': [], 'avg_y': 0})
    for y, row in enumerate(grid.values):
        for x, color in enumerate(row):
            if color != 0:
                colors[color]['cells'].append((x, y))
    
    for color, data in colors.items():
        data['avg_y'] = sum(y for _, y in data['cells']) / len(data['cells'])
    
    return colors

def identify_background_colors(grid: ColoredGrid) -> Tuple[int, int]:
    bottom_row = grid.values[-1]
    top_row = grid.values[0]
    bottom_bg = max(set(bottom_row), key=bottom_row.count) if any(bottom_row) else 0
    top_bg = max(set(top_row), key=top_row.count) if any(top_row) else 0
    return bottom_bg, top_bg

def copy_background_colors(output: ColoredGrid, input_grid: ColoredGrid, bottom_bg: int, top_bg: int):
    # Copy bottom background
    if bottom_bg:
        for y in range(len(input_grid.values) - 1, -1, -1):
            if input_grid.values[y].count(bottom_bg) / len(input_grid.values[y]) < 0.9:
                break
            output.values[y] = input_grid.values[y].copy()
    
    # Copy top background
    if top_bg:
        for y in range(len(input_grid.values)):
            if input_grid.values[y].count(top_bg) / len(input_grid.values[y]) < 0.9:
                break
            output.values[y] = input_grid.values[y].copy()

def sort_colors(colors: Dict[int, Dict], bottom_bg: int, top_bg: int) -> List[int]:
    return sorted([c for c in colors if c not in {bottom_bg, top_bg, 0}], key=lambda c: colors[c]['avg_y'])

def create_vertical_stack(output: ColoredGrid, colors: List[int], bottom_bg: int, top_bg: int):
    non_bg_start = next(y for y in range(len(output.values)) if output.values[y].count(0) == len(output.values[y]))
    non_bg_end = next(y for y in range(len(output.values) - 1, -1, -1) if output.values[y].count(0) == len(output.values[y]))
    
    stack_height = len(colors) * 4 - 1
    start_y = (non_bg_start + non_bg_end - stack_height) // 2
    
    for color in colors:
        for y in range(3):
            for x in range(3):
                output.values[start_y + y][13 + x] = color
        start_y += 4
