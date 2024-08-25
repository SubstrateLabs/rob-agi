from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional
import random
import math

def solve_fd096ab6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the fd096ab6 challenge by expanding color clusters into hook or L-shapes.
    
    The solution involves the following steps:
    1. Identify all non-blue color clusters in the grid.
    2. Sort clusters by size in descending order.
    3. For each cluster:
       a. Determine the target shape (L or hook) and size based on the original cluster size.
       b. Expand the cluster towards the target shape and size, prioritizing certain directions.
       c. Avoid collisions with other non-blue colors during expansion.
    4. Perform two phases of expansion to ensure proper shape formation.
    5. Refine shapes by smoothing edges and removing isolated cells.
    
    This approach ensures that larger clusters are handled first and that all clusters
    are expanded into consistent L or hook shapes while maintaining their relative positions.
    
    Args:
    input_grid (ColoredGrid): The initial grid state.
    
    Returns:
    ColoredGrid: The transformed grid with expanded color clusters in hook or L-shapes.
    """
    grid = input_grid.deep_copy()
    colors = set(range(2, 10))  # All colors except blue (1)
    
    for _ in range(2):  # Perform expansion in 2 phases
        clusters = []
        for color in colors:
            regions = grid.find_connected_regions(color)
            clusters.extend((color, region) for region in regions)
        
        clusters.sort(key=lambda x: len(x[1]), reverse=True)  # Sort by cluster size
        
        for color, cluster in clusters:
            expand_cluster_to_hook(grid, color, cluster)
    
    refine_shapes(grid)
    return grid

def expand_cluster_to_hook(grid: ColoredGrid, color: int, cluster: List[Tuple[int, int]]):
    cluster_size = len(cluster)
    target_size = calculate_target_size(cluster_size)
    expansion_cells = get_expansion_cells(grid, cluster)
    
    while len(cluster) < target_size and expansion_cells:
        best_cell = choose_best_expansion_cell(cluster, expansion_cells)
        if best_cell:
            r, c = best_cell
            grid.set_cell(r, c, color)
            cluster.append(best_cell)
            expansion_cells = get_expansion_cells(grid, cluster)
        else:
            break

def calculate_target_size(cluster_size: int) -> int:
    if cluster_size <= 3:
        return min(cluster_size * 2, 5)
    elif cluster_size <= 8:
        return min(int(cluster_size * 1.5), 12)
    else:
        return min(cluster_size + 4, 16)

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

def choose_best_expansion_cell(cluster: List[Tuple[int, int]], expansion_cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not expansion_cells:
        return None
    
    cluster_shape = analyze_cluster_shape(cluster)
    
    if cluster_shape == "single":
        return expansion_cells[0]
    elif cluster_shape == "linear":
        return choose_perpendicular_cell(cluster, expansion_cells)
    elif cluster_shape == "L":
        return choose_corner_cell(cluster, expansion_cells)
    else:
        return choose_edge_extension_cell(cluster, expansion_cells)

def analyze_cluster_shape(cluster: List[Tuple[int, int]]) -> str:
    if len(cluster) == 1:
        return "single"
    elif len(cluster) == 2:
        return "linear"
    elif len(cluster) == 3:
        return "L" if not is_linear(cluster) else "linear"
    else:
        return "complex"

def is_linear(cluster: List[Tuple[int, int]]) -> bool:
    if len(cluster) <= 2:
        return True
    points = sorted(cluster)
    return (points[0][0] == points[-1][0]) or (points[0][1] == points[-1][1])

def choose_perpendicular_cell(cluster: List[Tuple[int, int]], expansion_cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if is_linear(cluster):
        main_axis = 0 if cluster[0][0] == cluster[-1][0] else 1
        for cell in expansion_cells:
            if cell[main_axis] != cluster[0][main_axis]:
                return cell
    return expansion_cells[0] if expansion_cells else None

def choose_corner_cell(cluster: List[Tuple[int, int]], expansion_cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    corners = [c for c in expansion_cells if sum(abs(c[i] - cluster[0][i]) for i in range(2)) == 2]
    return corners[0] if corners else (expansion_cells[0] if expansion_cells else None)

def choose_edge_extension_cell(cluster: List[Tuple[int, int]], expansion_cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    edges = [c for c in expansion_cells if sum(abs(c[i] - cluster[0][i]) for i in range(2)) == 1]
    return edges[0] if edges else (expansion_cells[0] if expansion_cells else None)

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
