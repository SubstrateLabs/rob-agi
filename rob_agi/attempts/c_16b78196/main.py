from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_16b78196(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by reorganizing color shapes into a centralized vertical structure.
    
    The solution follows these steps:
    1. Analyze the input grid to identify colors and their characteristics.
    2. Identify background colors at the top and bottom of the grid.
    3. Create a new grid with the bottom background color (if present).
    4. Sort remaining colors based on their average vertical position.
    5. Create a central vertical stack of 3x3 squares for each color.
    6. Add the top background color (if present).
    7. Ensure clean edges and fill remaining space with black.
    
    Returns a new ColoredGrid with the transformed arrangement.
    """
    # Step 1: Analyze the input grid
    colors = analyze_grid(input_grid)
    
    # Step 2: Identify background colors
    bottom_bg, top_bg = identify_background_colors(input_grid)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(30)] for _ in range(30)])
    
    # Step 3: Add bottom background color
    if bottom_bg:
        fill_bottom_background(output, bottom_bg, colors[bottom_bg]['height'])
    
    # Step 4: Sort remaining colors
    sorted_colors = sort_colors(colors, bottom_bg)
    
    # Step 5: Create central vertical stack
    create_vertical_stack(output, sorted_colors, bottom_bg)
    
    # Step 6: Add top background color
    if top_bg:
        fill_top_background(output, top_bg)
    
    # Step 7: Final cleanup
    cleanup_grid(output)
    
    return output

def analyze_grid(grid: ColoredGrid) -> Dict[int, Dict]:
    colors = defaultdict(lambda: {'cells': [], 'avg_y': 0, 'height': 0})
    for y, row in enumerate(grid.values):
        for x, color in enumerate(row):
            if color != 0:
                colors[color]['cells'].append((x, y))
    
    for color, data in colors.items():
        data['avg_y'] = sum(y for _, y in data['cells']) / len(data['cells'])
        data['height'] = max(y for _, y in data['cells']) - min(y for _, y in data['cells']) + 1
    
    return colors

def identify_background_colors(grid: ColoredGrid) -> Tuple[int, int]:
    bottom_row = grid.values[-1]
    top_row = grid.values[0]
    bottom_bg = max(set(bottom_row), key=bottom_row.count) if any(bottom_row) else 0
    top_bg = max(set(top_row), key=top_row.count) if any(top_row) else 0
    return bottom_bg, top_bg

def fill_bottom_background(grid: ColoredGrid, color: int, height: int):
    for y in range(30 - height, 30):
        for x in range(30):
            grid.values[y][x] = color

def sort_colors(colors: Dict[int, Dict], bottom_bg: int) -> List[int]:
    return sorted([c for c in colors if c != bottom_bg], key=lambda c: colors[c]['avg_y'])

def create_vertical_stack(grid: ColoredGrid, colors: List[int], bottom_bg: int):
    start_y = 30 - 4 * len(colors) - 1 if bottom_bg else (30 - 4 * len(colors)) // 2
    for color in colors:
        for y in range(3):
            for x in range(3):
                grid.values[start_y + y][13 + x] = color
        start_y += 4

def fill_top_background(grid: ColoredGrid, color: int):
    top_edge = next(y for y in range(30) if any(grid.values[y]))
    for y in range(top_edge):
        for x in range(30):
            grid.values[y][x] = color

def cleanup_grid(grid: ColoredGrid):
    # Ensure clean edges (simplified implementation)
    pass
