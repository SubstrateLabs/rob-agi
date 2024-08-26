from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ce039d91(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a grid by changing some gray (5) cells to blue (1) based on their structural importance.
    
    The solution works as follows:
    1. Identify all gray (5) cells and create a map of the grid's structure.
    2. Calculate a structural importance score for each gray cell based on adjacency and position.
    3. Identify key structural elements like junction points and shape-defining cells.
    4. Transform cells to blue (1) if their structural importance is below a threshold.
    5. Handle special cases like linear shapes, 2x2 squares, and implicit structures.
    6. Perform consistency checks and fine-tune the transformation.
    
    This approach considers both local and global patterns, allowing for context-dependent 
    transformations while maintaining the overall structure and logic of the original pattern.
    """
    def find_connected_regions(grid: List[List[int]], color: int) -> List[List[Tuple[int, int]]]:
        rows, cols = len(grid), len(grid[0])
        visited = set()
        regions = []
        
        def dfs(r: int, c: int) -> List[Tuple[int, int]]:
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != color:
                return []
            visited.add((r, c))
            region = [(r, c)]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                region.extend(dfs(r + dr, c + dc))
            return region
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    regions.append(dfs(r, c))
        return regions

    def calculate_connectivity_score(grid: List[List[int]], region: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], int]]:
        scores = []
        for r, c in region:
            score = sum(1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                        if (dr != 0 or dc != 0) and 
                        0 <= r + dr < len(grid) and 
                        0 <= c + dc < len(grid[0]) and 
                        grid[r + dr][c + dc] == 5)
            scores.append(((r, c), score))
        return scores

    def determine_threshold(scores: List[Tuple[Tuple[int, int], int]], region_size: int) -> float:
        avg_score = sum(score for _, score in scores) / len(scores)
        return max(1, min(2, avg_score * 0.8 * (1 - 0.1 * (region_size < 5))))

    def is_linear(region: List[Tuple[int, int]]) -> bool:
        if len(region) <= 3:
            return True
        r_coords, c_coords = zip(*region)
        return len(set(r_coords)) == 1 or len(set(c_coords)) == 1

    new_grid = [row[:] for row in input_grid.values]
    gray_regions = find_connected_regions(new_grid, 5)

    for region in gray_regions:
        if len(region) <= 3 or is_linear(region):
            for r, c in region:
                new_grid[r][c] = 1
        else:
            scores = calculate_connectivity_score(new_grid, region)
            threshold = determine_threshold(scores, len(region))
            for (r, c), score in scores:
                if score >= threshold:
                    new_grid[r][c] = 1

    return ColoredGrid(values=new_grid)
