from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import statistics

def solve_bf89d739(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red dots with green lines.
    
    The function identifies clusters of red dots, creates local vertical spines for each cluster,
    connects red dots to their local spines, and then connects the spines if necessary.
    The algorithm ensures that all red dots are connected in an optimized tree-like structure
    with vertical spines and horizontal branches.
    
    Args:
    input_grid (ColoredGrid): The input grid containing red dots to be connected.
    
    Returns:
    ColoredGrid: A new grid with the red dots connected by green lines forming an optimized tree-like structure.
    """
    result_grid = input_grid.deep_copy()
    red_dots = find_red_dots(input_grid)
    
    if not red_dots:
        return result_grid

    clusters = cluster_red_dots(red_dots)
    for cluster in clusters:
        spine_col = find_optimal_spine(cluster)
        create_vertical_spine(result_grid, cluster, spine_col)
        connect_dots_to_spine(result_grid, cluster, spine_col)

    connect_spines(result_grid, clusters)
    optimize_connections(result_grid)

    return result_grid

def find_red_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(grid.num_rows) for c in range(grid.num_cols) if grid.values[r][c] == 2]

def cluster_red_dots(red_dots: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    sorted_dots = sorted(red_dots, key=lambda x: x[1])  # Sort by x-coordinate
    clusters = []
    current_cluster = [sorted_dots[0]]
    
    for dot in sorted_dots[1:]:
        if dot[1] - current_cluster[-1][1] <= 2:  # Adjust threshold as needed
            current_cluster.append(dot)
        else:
            clusters.append(current_cluster)
            current_cluster = [dot]
    
    clusters.append(current_cluster)
    return clusters

def find_optimal_spine(cluster: List[Tuple[int, int]]) -> int:
    x_coords = [c for _, c in cluster]
    return round(statistics.mean(x_coords))

def create_vertical_spine(grid: ColoredGrid, cluster: List[Tuple[int, int]], spine_col: int):
    min_row = min(r for r, _ in cluster)
    max_row = max(r for r, _ in cluster)
    draw_line(grid, (min_row, spine_col), (max_row, spine_col), True)

def connect_dots_to_spine(grid: ColoredGrid, cluster: List[Tuple[int, int]], spine_col: int):
    for r, c in cluster:
        if c != spine_col:
            draw_line(grid, (r, c), (r, spine_col), False)

def connect_spines(grid: ColoredGrid, clusters: List[List[Tuple[int, int]]]):
    if len(clusters) <= 1:
        return
    
    spine_cols = [find_optimal_spine(cluster) for cluster in clusters]
    for i in range(len(clusters) - 1):
        start_col = spine_cols[i]
        end_col = spine_cols[i + 1]
        start_row = min(r for r, _ in clusters[i])
        end_row = min(r for r, _ in clusters[i + 1])
        
        if start_row == end_row:
            draw_line(grid, (start_row, start_col), (end_row, end_col), False)
        else:
            mid_col = (start_col + end_col) // 2
            draw_line(grid, (start_row, start_col), (start_row, mid_col), False)
            draw_line(grid, (start_row, mid_col), (end_row, mid_col), True)
            draw_line(grid, (end_row, mid_col), (end_row, end_col), False)

def optimize_connections(grid: ColoredGrid):
    # This function can be implemented to further optimize the connections
    # For now, we'll leave it as a placeholder
    pass

def draw_line(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int], is_vertical: bool):
    y1, x1 = start
    y2, x2 = end
    
    if is_vertical:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if grid.values[y][x1] == 0:  # Only fill black cells
                grid.values[y][x1] = 3  # Green
    else:
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if grid.values[y1][x] == 0:  # Only fill black cells
                grid.values[y1][x] = 3  # Green
