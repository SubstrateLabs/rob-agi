from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_bd14c3bf(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by changing irregular blue shapes to red
    while preserving regular blue shapes. The function identifies connected regions
    of blue cells, assesses their regularity based on symmetry and shape characteristics,
    and changes the color of irregular shapes to red based on a regularity score threshold.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with irregular blue shapes changed to red.
    """
    output_grid = input_grid.deep_copy()
    blue_regions = find_connected_regions(output_grid, 1)  # 1 represents blue
    
    for region in blue_regions:
        regularity_score = calculate_regularity(region)
        if regularity_score < 0.6:  # Threshold determined by analyzing examples
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

def calculate_regularity(region: List[Tuple[int, int]]) -> float:
    """Calculate the regularity of a shape based on symmetry and shape characteristics."""
    if not region:
        return 0
    
    # Calculate bounding box
    min_r = min(r for r, _ in region)
    max_r = max(r for r, _ in region)
    min_c = min(c for _, c in region)
    max_c = max(c for _, c in region)
    
    # Create a binary representation of the shape
    shape = [[0 for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)]
    for r, c in region:
        shape[r - min_r][c - min_c] = 1
    
    # Check for symmetry
    vertical_symmetry = check_vertical_symmetry(shape)
    horizontal_symmetry = check_horizontal_symmetry(shape)
    
    # Count straight edges and corners
    straight_edges, corners = count_edges_and_corners(shape)
    
    # Calculate regularity score
    regularity = (vertical_symmetry + horizontal_symmetry + straight_edges / len(region) + corners / len(region)) / 4
    
    return regularity

def check_vertical_symmetry(shape: List[List[int]]) -> float:
    rows, cols = len(shape), len(shape[0])
    symmetry_score = 0
    for r in range(rows):
        for c in range(cols // 2):
            if shape[r][c] == shape[r][cols - 1 - c]:
                symmetry_score += 1
    return symmetry_score / (rows * cols // 2)

def check_horizontal_symmetry(shape: List[List[int]]) -> float:
    rows, cols = len(shape), len(shape[0])
    symmetry_score = 0
    for r in range(rows // 2):
        for c in range(cols):
            if shape[r][c] == shape[rows - 1 - r][c]:
                symmetry_score += 1
    return symmetry_score / (rows // 2 * cols)

def count_edges_and_corners(shape: List[List[int]]) -> Tuple[int, int]:
    rows, cols = len(shape), len(shape[0])
    straight_edges = 0
    corners = 0
    for r in range(rows):
        for c in range(cols):
            if shape[r][c] == 1:
                neighbors = sum(shape[r+dr][c+dc] for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)] 
                                if 0 <= r+dr < rows and 0 <= c+dc < cols)
                if neighbors == 2:
                    straight_edges += 1
                elif neighbors == 1:
                    corners += 1
    return straight_edges, corners
