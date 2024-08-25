from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Optional
import random
import math

def solve_fd096ab6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the fd096ab6 challenge by expanding color clusters into coherent shapes.
    
    The solution involves the following steps:
    1. Identify all non-blue color clusters in the grid.
    2. Sort clusters by size in descending order.
    3. For each cluster:
       a. Determine the target size based on the original cluster size.
       b. Analyze the cluster shape and available space.
       c. Expand the cluster towards the target size, prioritizing coherent shapes.
       d. Avoid collisions with other non-blue colors during expansion.
    4. Refine shapes by filling gaps and removing isolated cells.
    5. Perform a final pass to ensure all expansions are coherent.
    
    This approach ensures that clusters are expanded appropriately while maintaining
    their general position and creating visually coherent shapes.
    
    Args:
    input_grid (ColoredGrid): The initial grid state.
    
    Returns:
    ColoredGrid: The transformed grid with expanded color clusters.
    """
    grid = input_grid.deep_copy()
    colors = set(range(2, 10))  # All colors except blue (1)
    
    clusters = []
    for color in colors:
        regions = grid.find_connected_regions(color)
        clusters.extend((color, region) for region in regions)
    
    clusters.sort(key=lambda x: len(x[1]), reverse=True)  # Sort by cluster size
    
    for color, cluster in clusters:
        expand_cluster(grid, color, cluster)
    
    refine_shapes(grid)
    final_pass(grid)
    return grid

def expand_cluster(grid: ColoredGrid, color: int, cluster: List[Tuple[int, int]]):
    cluster_size = len(cluster)
    target_size = calculate_target_size(cluster_size)
    shape = analyze_cluster_shape(cluster)
    
    while len(cluster) < target_size:
        expansion_cells = get_expansion_cells(grid, cluster)
        if not expansion_cells:
            break
        
        best_cell = choose_best_expansion_cell(cluster, expansion_cells, shape)
        if best_cell:
            r, c = best_cell
            grid.set_cell(r, c, color)
            cluster.append(best_cell)
            shape = analyze_cluster_shape(cluster)
        else:
            break
    
    fill_gaps(grid, color, cluster)

def calculate_target_size(cluster_size: int) -> int:
    if cluster_size <= 3:
        return min(max(cluster_size * 2, 4), 6)
    elif cluster_size <= 8:
        return min(max(cluster_size * 1.5, 8), 12)
    else:
        return min(cluster_size * 1.25, 16)

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

def choose_best_expansion_cell(cluster: List[Tuple[int, int]], expansion_cells: List[Tuple[int, int]], shape: str) -> Optional[Tuple[int, int]]:
    if not expansion_cells:
        return None
    
    center_r, center_c = sum(r for r, _ in cluster) // len(cluster), sum(c for _, c in cluster) // len(cluster)
    
    if shape == "linear":
        return min(expansion_cells, key=lambda cell: abs(cell[0] - center_r) + abs(cell[1] - center_c))
    elif shape in ["horizontal", "vertical"]:
        return max(expansion_cells, key=lambda cell: abs(cell[0] - center_r) if shape == "horizontal" else abs(cell[1] - center_c))
    elif shape == "square":
        return min(expansion_cells, key=lambda cell: max(abs(cell[0] - center_r), abs(cell[1] - center_c)))
    else:  # irregular
        return min(expansion_cells, key=lambda cell: (cell[0] - center_r)**2 + (cell[1] - center_c)**2)

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
                remove_isolated_cell(grid, r, c)

def remove_isolated_cell(grid: ColoredGrid, r: int, c: int):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = grid.get_dimensions()
    color = grid.get_cell(r, c)
    same_color_neighbors = 0
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == color:
            same_color_neighbors += 1
    
    if same_color_neighbors == 0:
        grid.set_cell(r, c, 1)
def analyze_cluster_shape(cluster: List[Tuple[int, int]]) -> str:
    if len(cluster) <= 2:
        return "linear"
    
    min_r, min_c = min(cluster)
    max_r, max_c = max(cluster)
    width = max_c - min_c + 1
    height = max_r - min_r + 1
    
    if width > height * 2:
        return "horizontal"
    elif height > width * 2:
        return "vertical"
    elif abs(width - height) <= 1:
        return "square"
    else:
        return "irregular"
def fill_gaps(grid: ColoredGrid, color: int, cluster: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    cluster_set = set(cluster)
    
    for r, c in cluster:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 1:
                neighbors = sum(1 for d in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (nr+d[0], nc+d[1]) in cluster_set)
                if neighbors >= 2:
                    grid.set_cell(nr, nc, color)
                    cluster.append((nr, nc))
                    cluster_set.add((nr, nc))
def final_pass(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) != 1:
                ensure_coherent_shape(grid, r, c)

def ensure_coherent_shape(grid: ColoredGrid, r: int, c: int):
    color = grid.get_cell(r, c)
    cluster = grid.find_connected_regions(color)[0]
    if len(cluster) < 4:
        for rr, cc in cluster:
            grid.set_cell(rr, cc, 1)
    elif analyze_cluster_shape(cluster) == "irregular":
        expand_cluster(grid, color, cluster)
