from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random

def solve_6a11f6da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 15x5 input grid into a 5x5 output grid by analyzing patterns of blue (1),
    sky blue (8), and magenta (6) in three 5x5 sections of the input.
    
    1. Analyze color distributions and patterns in the input grid.
    2. Create a 5x5 output grid with a balanced color distribution.
    3. Place colors strategically to echo input patterns while maintaining balance.
    4. Ensure all four colors (including black) are present in the output.
    5. Maintain a color ratio of 9:5:3:2 (magenta:blue:sky blue:black).
    6. Refine the pattern to create visual interest and echo input patterns.
    
    Returns a 5x5 ColoredGrid preserving key patterns and color distributions.
    """
    # Analyze input
    blue_section = input_grid.extract_subgrid(0, 0, 5, 5)
    sky_section = input_grid.extract_subgrid(5, 0, 5, 5)
    magenta_section = input_grid.extract_subgrid(10, 0, 5, 5)
    
    blue_pattern = find_significant_pattern(blue_section, 1)
    sky_pattern = find_significant_pattern(sky_section, 8)
    magenta_pattern = find_significant_pattern(magenta_section, 6)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])
    
    # Place colors
    place_magenta(output, magenta_pattern)
    place_blue(output, blue_pattern)
    place_sky_blue(output, sky_pattern)
    place_black(output)
    
    # Balance colors
    balance_colors(output)
    
    # Refine pattern
    refine_pattern(output, blue_pattern, sky_pattern, magenta_pattern)
    
    return output

def find_significant_pattern(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    regions = grid.find_connected_regions(color)
    return max(regions, key=len) if regions else []

def place_magenta(grid: ColoredGrid, pattern: List[Tuple[int, int]]):
    shape = determine_shape(pattern)
    for x, y in shape:
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, 6)
        if grid.count_color(6) == 9:
            break

def determine_shape(pattern: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    shapes = {
        'U': [(0,0), (0,4), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)],
        'C': [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        'L': [(0,0), (1,0), (2,0), (3,0), (4,0), (4,1), (4,2), (4,3), (4,4)],
        'cross': [(0,2), (1,2), (2,0), (2,1), (2,2), (2,3), (2,4), (3,2), (4,2)]
    }
    return random.choice(list(shapes.values()))

def place_blue(grid: ColoredGrid, pattern: List[Tuple[int, int]]):
    corners = [(0,0), (0,4), (4,0), (4,4)]
    random.shuffle(corners)
    for x, y in corners + list(pattern):
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, 1)
        if grid.count_color(1) == 5:
            break

def place_sky_blue(grid: ColoredGrid, pattern: List[Tuple[int, int]]):
    edges = [(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)]
    random.shuffle(edges)
    for x, y in edges + list(pattern):
        if grid.get_cell(x, y) == 0:
            grid.set_cell(x, y, 8)
        if grid.count_color(8) == 3:
            break

def place_black(grid: ColoredGrid):
    empty_cells = [(x, y) for x in range(5) for y in range(5) if grid.get_cell(x, y) == 0]
    random.shuffle(empty_cells)
    for x, y in empty_cells:
        grid.set_cell(x, y, 0)
        if grid.count_color(0) == 2:
            break

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

def refine_pattern(grid: ColoredGrid, blue_pattern: List[Tuple[int, int]], sky_pattern: List[Tuple[int, int]], magenta_pattern: List[Tuple[int, int]]):
    # Ensure at least one corner is blue or sky blue
    corners = [(0,0), (0,4), (4,0), (4,4)]
    if all(grid.get_cell(x, y) not in [1, 8] for x, y in corners):
        x, y = random.choice(corners)
        nearest_blue = min((abs(x-bx) + abs(y-by), bx, by) for bx, by in blue_pattern + sky_pattern)
        grid.set_cell(x, y, grid.get_cell(nearest_blue[1], nearest_blue[2]))
        grid.set_cell(nearest_blue[1], nearest_blue[2], 6)  # Replace with magenta
    
    # Break up large color blocks
    for _ in range(3):  # Make a few refinements
        for x in range(5):
            for y in range(5):
                neighbors = [(x+dx, y+dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)] if 0 <= x+dx < 5 and 0 <= y+dy < 5]
                if all(grid.get_cell(nx, ny) == grid.get_cell(x, y) for nx, ny in neighbors):
                    different_color = random.choice([c for c in [6, 1, 8, 0] if c != grid.get_cell(x, y)])
                    grid.set_cell(x, y, different_color)
