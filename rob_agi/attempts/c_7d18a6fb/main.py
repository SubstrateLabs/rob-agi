from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_7d18a6fb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 7x7 output grid by identifying significant patterns and arranging them.
    
    1. Identifies connected regions of the same color in the input grid.
    2. Ranks patterns based on size, color rarity, and shape complexity.
    3. Selects top 4 patterns and compresses them into 3x3 representations.
    4. Places compressed patterns in quadrants of a 7x7 grid, maintaining relative positions.
    5. Fills the central cross with black (0) to separate quadrants.
    
    Returns a 7x7 ColoredGrid with the arranged patterns.
    """
    patterns = identify_patterns(input_grid)
    ranked_patterns = rank_patterns(patterns, input_grid)
    compressed_patterns = compress_patterns(ranked_patterns[:4])
    output_grid = create_output_grid(compressed_patterns, input_grid)
    return output_grid

def identify_patterns(grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]]]]:
    patterns = []
    visited = set()
    rows, cols = grid.get_dimensions()

    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid.values[r][c] != color:
            return []
        visited.add((r, c))
        region = [(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            region.extend(dfs(r + dr, c + dc, color))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.values[r][c] != 0:
                color = grid.values[r][c]
                region = dfs(r, c, color)
                if region:
                    patterns.append((color, region))

    return patterns

def rank_patterns(patterns: List[Tuple[int, List[Tuple[int, int]]]], grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]]]]:
    color_counts = defaultdict(int)
    for row in grid.values:
        for cell in row:
            if cell != 0:
                color_counts[cell] += 1

    def pattern_score(pattern):
        color, region = pattern
        size = len(region)
        rarity = 1 / color_counts[color]
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        complexity = len(region) / ((max_r - min_r + 1) * (max_c - min_c + 1))
        return size * rarity * complexity

    return sorted(patterns, key=pattern_score, reverse=True)

def compress_patterns(patterns: List[Tuple[int, List[Tuple[int, int]]]]) -> List[Tuple[int, List[List[int]]]]:
    compressed = []
    for color, region in patterns:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        
        compressed_pattern = [[0 for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                r_start = min_r + (max_r - min_r + 1) * r // 3
                r_end = min_r + (max_r - min_r + 1) * (r + 1) // 3
                c_start = min_c + (max_c - min_c + 1) * c // 3
                c_end = min_c + (max_c - min_c + 1) * (c + 1) // 3
                cells = [(rr, cc) for rr, cc in region if r_start <= rr <= r_end and c_start <= cc <= c_end]
                if cells:
                    compressed_pattern[r][c] = color
        
        compressed.append((color, compressed_pattern))
    
    return compressed

def create_output_grid(compressed_patterns: List[Tuple[int, List[List[int]]]], input_grid: ColoredGrid) -> ColoredGrid:
    output = [[0 for _ in range(7)] for _ in range(7)]
    input_rows, input_cols = input_grid.get_dimensions()
    quadrants = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    for (color, pattern), (qr, qc) in zip(compressed_patterns, quadrants):
        # Determine pattern position within quadrant
        pattern_r = qr + (3 if 2 * qr > input_rows else 0)
        pattern_c = qc + (3 if 2 * qc > input_cols else 0)
        
        for r in range(3):
            for c in range(3):
                output[pattern_r + r][pattern_c + c] = pattern[r][c]
    
    # Fill central cross
    for i in range(7):
        output[3][i] = 0
        output[i][3] = 0
    
    return ColoredGrid(values=output)
