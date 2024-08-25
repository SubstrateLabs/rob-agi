from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bd14c3bf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by changing complex blue shapes to red
    while preserving simpler blue shapes. The function identifies connected regions
    of blue cells, calculates their complexity, and changes the color of complex
    shapes to red based on a threshold.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with complex blue shapes changed to red.
    """
    output_grid = input_grid.deep_copy()
    blue_regions = find_connected_regions(output_grid, 1)  # 1 represents blue
    
    for region in blue_regions:
        complexity = calculate_complexity(region)
        if complexity > 10:  # Threshold determined by analyzing examples
            for r, c in region:
                output_grid.set_cell(r, c, 2)  # 2 represents red
    
    return output_grid

def find_connected_regions(grid: ColoredGrid, color: int) -> List[List[Tuple[int, int]]]:
    """Find all connected regions of a specific color in the grid."""
    rows, cols = grid.get_dimensions()
    visited = set()
    regions = []
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == color and (r, c) not in visited:
                region = []
                stack = [(r, c)]
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                        visited.add((curr_r, curr_c))
                        region.append((curr_r, curr_c))
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                stack.append((nr, nc))
                regions.append(region)
    
    return regions

def calculate_complexity(region: List[Tuple[int, int]]) -> float:
    """Calculate the complexity of a shape based on its size and perimeter."""
    if not region:
        return 0
    
    # Calculate bounding box
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    # Calculate area and perimeter
    area = len(region)
    perimeter = sum(1 for r, c in region if (r+1, c) not in region or
                                           (r-1, c) not in region or
                                           (r, c+1) not in region or
                                           (r, c-1) not in region)
    
    # Calculate complexity score
    bounding_box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
    complexity = (perimeter * bounding_box_area) / (area * area)
    
    return complexity
