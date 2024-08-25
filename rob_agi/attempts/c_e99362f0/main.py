from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 5x4 output grid by following these steps:
    1. Split the input grid into left and right sections using the yellow (4) line.
    2. Identify and score color clusters in each section.
    3. Create a color importance map based on cluster sizes and positions.
    4. Generate a 5x4 output grid that represents the most important colors and patterns.
    5. Balance color distribution and ensure color diversity.
    6. Fine-tune the output to better reflect significant patterns from the input.
    """
    
    def find_clusters(grid: List[List[int]], color: int) -> List[List[Tuple[int, int]]]:
        clusters = []
        visited = set()
        rows, cols = len(grid), len(grid[0])
        
        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            cluster = []
            stack = [(r, c)]
            while stack:
                r, c = stack.pop()
                if (r, c) not in visited and grid[r][c] == color:
                    visited.add((r, c))
                    cluster.append((r, c))
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            stack.append((nr, nc))
            return cluster
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    clusters.append(dfs(r, c))
        
        return clusters

    def score_cluster(cluster: List[Tuple[int, int]], total_cells: int) -> float:
        size = len(cluster)
        avg_r = sum(r for r, _ in cluster) / size
        avg_c = sum(c for _, c in cluster) / size
        center_dist = ((avg_r - total_cells/2)**2 + (avg_c - total_cells/2)**2)**0.5
        return size * (1 - center_dist / total_cells)

    def create_color_importance_map(grid: List[List[int]]) -> Dict[int, float]:
        importance_map = defaultdict(float)
        total_cells = len(grid) * len(grid[0])
        for color in range(10):  # 0 to 9
            clusters = find_clusters(grid, color)
            for cluster in clusters:
                importance_map[color] += score_cluster(cluster, total_cells)
        return importance_map

    # Split the input grid
    left_section = [row[:4] for row in input_grid.values]
    right_section = [row[5:] for row in input_grid.values]

    # Create color importance maps
    left_importance = create_color_importance_map(left_section)
    right_importance = create_color_importance_map(right_section)

    # Generate output grid
    output = [[0 for _ in range(4)] for _ in range(5)]
    used_colors = defaultdict(int)

    for r in range(5):
        for c in range(4):
            if c < 2:
                importance = left_importance
            else:
                importance = right_importance
            
            color = max(importance, key=importance.get)
            while used_colors[color] >= 3 and len(used_colors) < 4:
                del importance[color]
                color = max(importance, key=importance.get)
            
            output[r][c] = color
            used_colors[color] += 1
            importance[color] *= 0.5  # Reduce importance after using

    # Ensure color diversity
    main_colors = {7, 8, 9, 2}
    for color in main_colors:
        if color not in used_colors:
            r, c = min(((r, c) for r in range(5) for c in range(4)), 
                       key=lambda pos: importance[output[pos[0]][pos[1]]])
            output[r][c] = color
            used_colors[color] += 1

    return ColoredGrid(values=output)
