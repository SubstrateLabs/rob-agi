from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import random
import math

def solve_fd096ab6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the fd096ab6 challenge by expanding color clusters asymmetrically.
    
    The solution involves the following steps:
    1. Identify all non-blue color clusters in the grid.
    2. Analyze each cluster's size, shape, and surrounding space.
    3. Determine expansion strategy based on cluster characteristics.
    4. Expand clusters iteratively, with size-based growth limits.
    5. Resolve conflicts between expanding clusters.
    6. Refine shapes and maintain consistency across similar clusters.
    7. Perform final validation and adjustments.
    
    Args:
    input_grid (ColoredGrid): The initial grid state.
    
    Returns:
    ColoredGrid: The transformed grid with expanded color clusters.
    """
    grid = input_grid.deep_copy()
    colors = set(range(2, 10))  # All colors except blue (1)
    
    for _ in range(3):  # Perform expansion in 3 phases
        clusters = []
        for color in colors:
            regions = grid.find_connected_regions(color)
            clusters.extend((color, region) for region in regions)
        
        clusters.sort(key=lambda x: len(x[1]), reverse=True)  # Sort by cluster size
        
        for color, cluster in clusters:
            expand_cluster(grid, color, cluster)
    
    refine_shapes(grid)
    return grid

def expand_cluster(grid: ColoredGrid, color: int, cluster: List[Tuple[int, int]]):
    cluster_size = len(cluster)
    growth_limit = calculate_growth_limit(cluster_size)
    expansion_cells = get_expansion_cells(grid, cluster)
    
    center = calculate_center(cluster)
    primary_direction = determine_primary_direction(cluster, center)
    
    expansion_cells = sorted(expansion_cells, key=lambda cell: expansion_priority(cell, center, primary_direction))
    
    for i, (r, c) in enumerate(expansion_cells):
        if i >= growth_limit:
            break
        if random.random() < expansion_probability(i, growth_limit):
            grid.set_cell(r, c, color)

def calculate_growth_limit(cluster_size: int) -> int:
    if cluster_size <= 3:
        return cluster_size * 2
    elif cluster_size <= 8:
        return int(cluster_size * 1.5)
    else:
        return cluster_size

def get_expansion_cells(grid: ColoredGrid, cluster: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    rows, cols = grid.get_dimensions()
    expansion_cells = set()
    
    for r, c in cluster:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 1:
                expansion_cells.add((nr, nc))
    
    return list(expansion_cells)

def calculate_center(cluster: List[Tuple[int, int]]) -> Tuple[float, float]:
    return (sum(r for r, _ in cluster) / len(cluster),
            sum(c for _, c in cluster) / len(cluster))

def determine_primary_direction(cluster: List[Tuple[int, int]], center: Tuple[float, float]) -> Tuple[float, float]:
    max_distance = 0
    primary_direction = (0, 0)
    
    for r, c in cluster:
        distance = math.sqrt((r - center[0])**2 + (c - center[1])**2)
        if distance > max_distance:
            max_distance = distance
            primary_direction = (center[0] - r, center[1] - c)
    
    magnitude = math.sqrt(primary_direction[0]**2 + primary_direction[1]**2)
    return (primary_direction[0] / magnitude, primary_direction[1] / magnitude)

def expansion_priority(cell: Tuple[int, int], center: Tuple[float, float], primary_direction: Tuple[float, float]) -> float:
    cell_direction = (center[0] - cell[0], center[1] - cell[1])
    dot_product = cell_direction[0] * primary_direction[0] + cell_direction[1] * primary_direction[1]
    return -dot_product  # Negative to prioritize cells in the primary direction

def expansion_probability(index: int, growth_limit: int) -> float:
    return 1 - (index / growth_limit)**0.5

def refine_shapes(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 1:
                smooth_cell(grid, r, c)

def smooth_cell(grid: ColoredGrid, r: int, c: int):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = grid.get_dimensions()
    color = grid.get_cell(r, c)
    blue_neighbors = 0
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 1:
            blue_neighbors += 1
    
    if blue_neighbors >= 3:
        grid.set_cell(r, c, 1)
    elif blue_neighbors == 0:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 1:
                grid.set_cell(nr, nc, color)
                break
