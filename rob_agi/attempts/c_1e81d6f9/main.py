from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_1e81d6f9(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by preserving the gray T-shape and intelligently limiting other colors.
    
    1. Preserves the T-shaped gray object.
    2. Analyzes connected regions of each color.
    3. Prioritizes larger connected regions while limiting each color to around 3 occurrences.
    4. Removes isolated color cells if necessary.
    
    Returns a new ColoredGrid with the transformed grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Step 1: Preserve the T-shaped gray object
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 5:  # Gray color
                output_grid.values[r][c] = 5
    
    def get_connected_region(r, c, color):
        region = []
        stack = [(r, c)]
        visited = set()
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                visited.add((curr_r, curr_c))
                region.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        stack.append((new_r, new_c))
        return region
    
    # Step 2-4: Process other colors
    for color in range(1, 10):
        if color == 5:  # Skip gray
            continue
        color_regions = []
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color and output_grid.values[r][c] == 0:
                    region = get_connected_region(r, c, color)
                    if region:
                        color_regions.append(region)
        
        # Sort regions by size, largest first
        color_regions.sort(key=len, reverse=True)
        
        count = 0
        for region in color_regions:
            if count + len(region) <= 3 or (count < 3 and len(region) > 1):
                for r, c in region:
                    output_grid.values[r][c] = color
                    count += 1
            if count >= 3:
                break
        
        # Add isolated cells if count is still less than 3
        if count < 3:
            for region in color_regions:
                if len(region) == 1:
                    r, c = region[0]
                    output_grid.values[r][c] = color
                    count += 1
                    if count == 3:
                        break
    
    return output_grid
