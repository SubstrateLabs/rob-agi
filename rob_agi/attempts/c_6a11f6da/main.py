from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_6a11f6da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 15x5 input grid into a 5x5 output grid by analyzing patterns of blue (1),
    sky blue (8), and magenta (6) in three 5x5 sections of the input.
    
    1. Split the input into three 5x5 sections.
    2. Analyze patterns in each section.
    3. Create a 5x5 output grid, prioritizing:
       a) Magenta patterns (bottom/right)
       b) Blue patterns (top-left)
       c) Sky blue (corners/edges)
    4. Fill remaining cells and balance colors.
    5. Ensure all three colors are present in the output.
    
    Returns a 5x5 ColoredGrid preserving key patterns and color distributions.
    """
    # Split input into three 5x5 sections
    blue_section = input_grid.extract_subgrid(0, 0, 5, 5)
    sky_section = input_grid.extract_subgrid(5, 0, 5, 5)
    magenta_section = input_grid.extract_subgrid(10, 0, 5, 5)
    
    # Analyze patterns
    blue_pattern = find_significant_pattern(blue_section, 1)
    sky_pattern = find_significant_pattern(sky_section, 8)
    magenta_pattern = find_significant_pattern(magenta_section, 6)
    
    # Create output grid
    output = ColoredGrid(values=[[0 for _ in range(5)] for _ in range(5)])
    
    # Place patterns
    place_pattern(output, magenta_pattern, 6, priority_area="bottom_right")
    place_pattern(output, blue_pattern, 1, priority_area="top_left")
    place_pattern(output, sky_pattern, 8, priority_area="corners")
    
    # Fill remaining cells and balance colors
    fill_and_balance(output)
    
    return output

def find_significant_pattern(grid: ColoredGrid, color: int) -> List[Tuple[int, int]]:
    regions = grid.find_connected_regions(color)
    return max(regions, key=len) if regions else []

def place_pattern(grid: ColoredGrid, pattern: List[Tuple[int, int]], color: int, priority_area: str):
    if not pattern:
        return
    
    if priority_area == "bottom_right":
        offset_x, offset_y = max(0, 5 - max(x for x, _ in pattern)), max(0, 5 - max(y for _, y in pattern))
    elif priority_area == "top_left":
        offset_x, offset_y = 0, 0
    else:  # corners
        offset_x, offset_y = 0, 0
    
    for x, y in pattern:
        new_x, new_y = x + offset_x, y + offset_y
        if 0 <= new_x < 5 and 0 <= new_y < 5:
            grid.set_cell(new_x, new_y, color)

def fill_and_balance(grid: ColoredGrid):
    colors = [1, 6, 8]
    for x in range(5):
        for y in range(5):
            if grid.get_cell(x, y) == 0:
                grid.set_cell(x, y, 6 if x + y > 4 else 1)
    
    # Ensure all colors are present
    for color in colors:
        if grid.count_color(color) == 0:
            for x in range(5):
                for y in range(5):
                    if grid.get_cell(x, y) != color:
                        grid.set_cell(x, y, color)
                        break
                if grid.count_color(color) > 0:
                    break
