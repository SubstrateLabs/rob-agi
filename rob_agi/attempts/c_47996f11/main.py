from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter, defaultdict

def solve_47996f11(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by removing the magenta region and generating coherent patterns.
    
    The solution involves:
    1. Identifying magenta regions in the grid
    2. Analyzing surrounding patterns and color distributions
    3. Extending existing patterns into magenta regions
    4. Ensuring vertical and horizontal continuity
    5. Balancing color distribution
    6. Refining edges and iteratively improving the result
    
    This approach aims to seamlessly integrate new patterns with the existing structure,
    maintaining the overall style and complexity of the original grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    
    def create_magenta_mask() -> List[List[bool]]:
        return [[input_grid.values[r][c] == 6 for c in range(cols)] for r in range(rows)]
    
    def analyze_grid() -> Dict:
        color_freq = Counter(color for row in input_grid.values for color in row if color != 6)
        transitions = defaultdict(Counter)
        patterns = defaultdict(Counter)
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] != 6:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] != 6:
                            transitions[input_grid.values[r][c]][input_grid.values[nr][nc]] += 1
                    if c < cols - 2:
                        pattern = tuple(input_grid.values[r][c:c+3])
                        patterns[pattern[0]][pattern[1:]] += 1
        return {"freq": color_freq, "transitions": transitions, "patterns": patterns}
    
    def extend_pattern(r: int, c: int, stats: Dict) -> int:
        neighbors = [output_grid.values[r+dr][c+dc] for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                     if 0 <= r+dr < rows and 0 <= c+dc < cols and not magenta_mask[r+dr][c+dc]]
        if not neighbors:
            return stats["freq"].most_common(1)[0][0]
        
        prev_color = max(set(neighbors), key=neighbors.count)
        if c >= 2 and not magenta_mask[r][c-2] and not magenta_mask[r][c-1]:
            pattern = (output_grid.values[r][c-2], output_grid.values[r][c-1])
            if pattern in stats["patterns"][prev_color]:
                return max(stats["patterns"][prev_color][pattern], key=stats["patterns"][prev_color][pattern].get)
        
        return max(stats["transitions"][prev_color], key=stats["transitions"][prev_color].get)
    
    def balance_color_distribution():
        target_dist = {k: v for k, v in grid_stats["freq"].items()}
        current_dist = Counter(color for row in output_grid.values for color in row)
        for r in range(rows):
            for c in range(cols):
                if magenta_mask[r][c]:
                    current_color = output_grid.values[r][c]
                    if current_dist[current_color] > target_dist[current_color]:
                        new_color = min(target_dist, key=lambda x: current_dist[x] / target_dist[x])
                        output_grid.values[r][c] = new_color
                        current_dist[current_color] -= 1
                        current_dist[new_color] += 1
    
    magenta_mask = create_magenta_mask()
    grid_stats = analyze_grid()
    
    # Fill magenta regions
    for _ in range(2):  # Two passes for better pattern extension
        for r in range(rows):
            for c in range(cols):
                if magenta_mask[r][c]:
                    output_grid.values[r][c] = extend_pattern(r, c, grid_stats)
    
    # Balance color distribution
    balance_color_distribution()
    
    # Refine edges
    for r in range(rows):
        for c in range(cols):
            if magenta_mask[r][c]:
                neighbors = [output_grid.values[r+dr][c+dc] for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                             if 0 <= r+dr < rows and 0 <= c+dc < cols and (dr, dc) != (0, 0)]
                output_grid.values[r][c] = max(set(neighbors), key=neighbors.count)
    
    return output_grid
