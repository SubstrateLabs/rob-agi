from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_6a11f6da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 15x5 input grid into a 5x5 output grid by analyzing patterns of blue (1),
    sky blue (8), and magenta (6) in three 5x5 sections of the input.
    
    1. Analyze color distributions and patterns in each section of the input grid.
    2. Determine dominant shapes and patterns for each color.
    3. Create a 5x5 output grid with a balanced color distribution.
    4. Place colors strategically to echo input patterns while maintaining balance.
    5. Ensure all four colors (including black) are present in the output.
    6. Maintain a color ratio of 9:5:3:2 (magenta:blue:sky blue:black).
    7. Refine the pattern to create visual interest and echo input patterns.
    
    Returns a 5x5 ColoredGrid preserving key patterns and color distributions.
    """
    # Analyze input
    blue_section = input_grid.extract_subgrid(0, 0, 5, 5)
    sky_section = input_grid.extract_subgrid(5, 0, 5, 5)
    magenta_section = input_grid.extract_subgrid(10, 0, 5, 5)
    
    blue_pattern = analyze_pattern(blue_section, 1)
    sky_pattern = analyze_pattern(sky_section, 8)
    magenta_pattern = analyze_pattern(magenta_section, 6)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])
    
    # Place colors
    place_dominant_color(output, magenta_pattern, 6, 9)
    place_secondary_color(output, blue_pattern, 1, 5)
    place_tertiary_color(output, sky_pattern, 8, 3)
    
    # Balance colors and refine pattern
    balance_colors(output)
    refine_pattern(output, blue_pattern, sky_pattern, magenta_pattern)
    
    return output

def analyze_pattern(grid: ColoredGrid, color: int) -> Dict[str, Any]:
    regions = grid.find_connected_regions(color)
    largest_region = max(regions, key=len) if regions else []
    
    pattern = {
        'largest_region': largest_region,
        'count': grid.count_color(color),
        'distribution': [sum(row.count(color) for row in grid.values[i:i+2]) for i in range(0, 5, 2)],
        'corners': sum(1 for x, y in [(0,0), (0,4), (4,0), (4,4)] if grid.get_cell(x, y) == color),
        'center': grid.get_cell(2, 2) == color
    }
    return pattern

def place_dominant_color(grid: ColoredGrid, pattern: Dict[str, Any], color: int, target: int):
    shape = determine_shape(pattern['largest_region'], pattern['distribution'])
    for x, y in shape:
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, color)
        if grid.count_color(color) == target:
            break

def place_secondary_color(grid: ColoredGrid, pattern: Dict[str, Any], color: int, target: int):
    corners = [(0,0), (0,4), (4,0), (4,4)]
    random.shuffle(corners)
    for x, y in corners + pattern['largest_region']:
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, color)
        if grid.count_color(color) == target:
            break

def place_tertiary_color(grid: ColoredGrid, pattern: Dict[str, Any], color: int, target: int):
    edges = [(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)]
    random.shuffle(edges)
    for x, y in edges + pattern['largest_region']:
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, color)
        if grid.count_color(color) == target:
            break

def determine_shape(region: List[Tuple[int, int]], distribution: List[int]) -> List[Tuple[int, int]]:
    shapes = {
        'U': [(0,0), (0,4), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)],
        'C': [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        'L': [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        'T': [(0,0), (0,1), (0,2), (0,3), (0,4), (1,2), (2,2), (3,2), (4,2)]
    }
    
    if len(region) >= 7 and distribution[0] > distribution[2]:
        return shapes['U']
    elif len(region) >= 7 and distribution[2] > distribution[0]:
        return shapes['C']
    elif len(region) >= 5 and distribution[1] > distribution[0] and distribution[1] > distribution[2]:
        return shapes['T']
    else:
        return shapes['L']

def balance_colors(grid: ColoredGrid):
    target_counts = {6: 9, 1: 5, 8: 3, 0: 2}
    colors = [6, 1, 8, 0]
    for color in colors:
        while grid.count_color(color) < target_counts[color]:
            for x in range(5):
                for y in range(5):
                    current_color = grid.get_cell(x, y)
                    if current_color != color and grid.count_color(current_color) > target_counts[current_color]:
                        grid.set_cell(x, y, color)
                        break
                if grid.count_color(color) == target_counts[color]:
                    break

def refine_pattern(grid: ColoredGrid, blue_pattern: Dict[str, Any], sky_pattern: Dict[str, Any], magenta_pattern: Dict[str, Any]):
    # Ensure at least one corner is blue or sky blue
    corners = [(0,0), (0,4), (4,0), (4,4)]
    if all(grid.get_cell(x, y) not in [1, 8] for x, y in corners):
        x, y = random.choice(corners)
        grid.set_cell(x, y, 1 if blue_pattern['corners'] > sky_pattern['corners'] else 8)
    
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
    for _ in range(2):  # Make a few refinements
        for x in range(5):
            for y in range(5):
                neighbors = [(x+dx, y+dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)] if 0 <= x+dx < 5 and 0 <= y+dy < 5]
                if len(set(grid.get_cell(nx, ny) for nx, ny in neighbors)) == 1:
                    different_color = random.choice([c for c in [6, 1, 8, 0] if c != grid.get_cell(x, y)])
                    grid.set_cell(x, y, different_color)
