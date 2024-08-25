from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_7d18a6fb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 7x7 output grid by identifying and arranging color patterns.
    
    1. Identifies the largest contiguous rectangle of a single color and excludes it.
    2. Finds unique color patterns in the remaining grid.
    3. Selects up to 4 patterns, prioritizing color diversity.
    4. Creates 3x3 representations of selected patterns.
    5. Places patterns in quadrants of a 7x7 grid based on their original positions.
    6. Fills the central cross with black (0) to separate quadrants.
    
    Returns a 7x7 ColoredGrid with the arranged patterns.
    """
    patterns = identify_patterns(input_grid)
    selected_patterns = select_patterns(patterns)
    compressed_patterns = compress_patterns(selected_patterns)
    output_grid = create_output_grid(compressed_patterns, input_grid)
    return output_grid

def identify_patterns(grid: ColoredGrid) -> List[Tuple[int, List[Tuple[int, int]]]]:
    patterns = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    # Find the largest rectangle
    largest_rect = find_largest_rectangle(grid)
    if largest_rect:
        color, rect_cells = largest_rect
        visited.update(rect_cells)
    
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

def find_largest_rectangle(grid: ColoredGrid) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    rows, cols = grid.get_dimensions()
    largest_area = 0
    largest_rect = None
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] != 0:
                color = grid.values[r][c]
                max_width = cols - c
                max_height = rows - r
                for width in range(1, max_width + 1):
                    for height in range(1, max_height + 1):
                        if all(grid.values[r + i][c + j] == color 
                               for i in range(height) for j in range(width)):
                            area = width * height
                            if area > largest_area:
                                largest_area = area
                                largest_rect = (color, [(r + i, c + j) 
                                                        for i in range(height) 
                                                        for j in range(width)])
    
    return largest_rect

def select_patterns(patterns: List[Tuple[int, List[Tuple[int, int]]]]) -> List[Tuple[int, List[Tuple[int, int]]]]:
    unique_colors = set()
    selected = []
    
    for color, region in patterns:
        if color not in unique_colors and len(selected) < 4:
            unique_colors.add(color)
            selected.append((color, region))
    
    return selected

def compress_patterns(patterns: List[Tuple[int, List[Tuple[int, int]]]]) -> List[Tuple[int, List[List[int]], Tuple[int, int]]]:
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
        
        center = ((min_r + max_r) // 2, (min_c + max_c) // 2)
        compressed.append((color, compressed_pattern, center))
    
    return compressed

def create_output_grid(compressed_patterns: List[Tuple[int, List[List[int]], Tuple[int, int]]], input_grid: ColoredGrid) -> ColoredGrid:
    output = [[0 for _ in range(7)] for _ in range(7)]
    input_rows, input_cols = input_grid.get_dimensions()
    quadrants = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    # Sort patterns based on their center position
    compressed_patterns.sort(key=lambda x: x[2])
    
    for (color, pattern, _), (qr, qc) in zip(compressed_patterns, quadrants):
        for r in range(3):
            for c in range(3):
                output[qr + r][qc + c] = pattern[r][c]
    
    # Fill central cross
    for i in range(7):
        output[3][i] = 0
        output[i][3] = 0
    
    return ColoredGrid(values=output)
