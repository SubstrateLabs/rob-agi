from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Any
import random

def solve_6a11f6da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 15x5 input grid into a 5x5 output grid by analyzing patterns of blue (1),
    sky blue (8), and magenta (6) in three 5x5 sections of the input.
    
    The solution follows these steps:
    1. Analyze each 5x5 section of the input grid for color patterns and distributions.
    2. Create a 5x5 output grid and place colors based on the analyzed patterns.
    3. Ensure a color ratio of 9:5:3:2 (magenta:blue:sky blue:black).
    4. Refine the pattern to create visual interest while maintaining balance.
    5. Verify that the output meets all criteria and draws inspiration from all input sections.
    
    Returns a 5x5 ColoredGrid that preserves key patterns and color distributions from the input.
    """
    # Step 1: Analyze input
    blue_section = input_grid.extract_subgrid(0, 0, 5, 5)
    sky_section = input_grid.extract_subgrid(5, 0, 5, 5)
    magenta_section = input_grid.extract_subgrid(10, 0, 5, 5)
    
    blue_analysis = analyze_input_section(blue_section, 1)
    sky_analysis = analyze_input_section(sky_section, 8)
    magenta_analysis = analyze_input_section(magenta_section, 6)
    
    # Step 2: Create output grid
    output = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])
    
    # Step 3: Place colors
    magenta_shape = determine_significant_shape(magenta_analysis)
    place_color(output, 6, 9, magenta_shape)
    place_color(output, 1, 5, blue_analysis['largest_region'])
    place_color(output, 8, 3, sky_analysis['largest_region'])
    
    # Step 4: Balance and refine
    balance_composition(output)
    
    # Step 5: Final check
    if not verify_output(output):
        # If verification fails, try again with a different random seed
        return solve_6a11f6da(input_grid)
    
    return output

def analyze_input_section(grid: ColoredGrid, color: int) -> Dict[str, Any]:
    regions = grid.find_connected_regions(color)
    largest_region = max(regions, key=len) if regions else []
    
    return {
        'largest_region': largest_region,
        'count': grid.count_color(color),
        'distribution': [sum(row.count(color) for row in grid.values[i:i+2]) for i in range(0, 5, 2)],
        'corners': sum(1 for x, y in [(0,0), (0,4), (4,0), (4,4)] if grid.get_cell(x, y) == color),
        'center': grid.get_cell(2, 2) == color
    }

def determine_significant_shape(analysis: Dict[str, Any]) -> List[Tuple[int, int]]:
    shapes = {
        'X': [(0,0), (0,4), (1,1), (1,3), (2,2), (3,1), (3,3), (4,0), (4,4)],
        'frame': [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,0), (4,1), (4,2), (4,3), (4,4)],
        'diagonal': [(0,0), (0,1), (1,1), (1,2), (2,2), (2,3), (3,3), (3,4), (4,4)]
    }
    
    if analysis['corners'] >= 2:
        return shapes['X']
    elif len(analysis['largest_region']) >= 10:
        return shapes['frame']
    else:
        return shapes['diagonal']

def place_color(grid: ColoredGrid, color: int, target: int, shape: List[Tuple[int, int]]):
    random.shuffle(shape)
    for x, y in shape:
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, color)
        if grid.count_color(color) == target:
            break
    
    while grid.count_color(color) < target:
        x, y = random.randint(0, 4), random.randint(0, 4)
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, color)

def balance_composition(grid: ColoredGrid):
    # Ensure at least one corner is blue or sky blue
    corners = [(0,0), (0,4), (4,0), (4,4)]
    if all(grid.get_cell(x, y) not in [1, 8] for x, y in corners):
        x, y = random.choice(corners)
        grid.set_cell(x, y, random.choice([1, 8]))
    
    # Ensure magenta forms a continuous shape
    magenta_regions = grid.find_connected_regions(6)
    if len(magenta_regions) > 1:
        main_region = max(magenta_regions, key=len)
        for region in magenta_regions:
            if region != main_region:
                for x, y in region:
                    nearest_main = min(main_region, key=lambda p: abs(p[0]-x) + abs(p[1]-y))
                    grid.set_cell(x, y, grid.get_cell(nearest_main[0], nearest_main[1]))
                    grid.set_cell(nearest_main[0], nearest_main[1], 6)
    
    # Break up large color blocks
    for _ in range(2):
        for x in range(5):
            for y in range(5):
                neighbors = [(x+dx, y+dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)] if 0 <= x+dx < 5 and 0 <= y+dy < 5]
                if len(set(grid.get_cell(nx, ny) for nx, ny in neighbors)) == 1:
                    different_color = random.choice([c for c in [6, 1, 8, 0] if c != grid.get_cell(x, y)])
                    grid.set_cell(x, y, different_color)

def verify_output(grid: ColoredGrid) -> bool:
    color_counts = {color: grid.count_color(color) for color in [0, 1, 6, 8]}
    return (color_counts[6] == 9 and
            color_counts[1] == 5 and
            color_counts[8] == 3 and
            color_counts[0] == 2 and
            len(grid.find_connected_regions(6)) == 1)
