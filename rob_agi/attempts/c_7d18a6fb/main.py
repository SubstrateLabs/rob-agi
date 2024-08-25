from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

def solve_7d18a6fb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 7x7 output grid by identifying and arranging color patterns.
    
    1. Preprocesses the input grid by identifying and excluding the largest contiguous rectangle of a single color.
    2. Divides the remaining grid into four quadrants.
    3. For each quadrant:
       a. Identifies distinct color patterns using a flood fill algorithm.
       b. Selects the most prominent pattern, prioritizing unique colors.
    4. Compresses each selected pattern into a 3x3 representation.
    5. Creates a 7x7 output grid:
       a. Places compressed patterns in the corners corresponding to their original quadrants.
       b. Fills the central cross with black (0) to separate quadrants.
    
    Returns a 7x7 ColoredGrid with the arranged patterns.
    """
    preprocessed_grid = preprocess_grid(input_grid)
    quadrants = divide_into_quadrants(preprocessed_grid)
    patterns = [process_quadrant(quadrant) for quadrant in quadrants]
    compressed_patterns = [compress_pattern(pattern) for pattern in patterns if pattern]
    output_grid = create_output_grid(compressed_patterns)
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

def create_output_grid(compressed_patterns: List[List[List[int]]]) -> ColoredGrid:
    output = [[0 for _ in range(7)] for _ in range(7)]
    quadrants = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    for pattern, (qr, qc) in zip(compressed_patterns, quadrants):
        for r in range(3):
            for c in range(3):
                output[qr + r][qc + c] = pattern[r][c]
    
    # Fill central cross
    for i in range(7):
        output[3][i] = 0
        output[i][3] = 0
    
    return ColoredGrid(values=output)
def preprocess_grid(grid: ColoredGrid) -> ColoredGrid:
    largest_rect = find_largest_rectangle(grid)
    if largest_rect:
        color, rect_cells = largest_rect
        new_grid = grid.deep_copy()
        for r, c in rect_cells:
            new_grid.values[r][c] = 0  # Set to black (excluded)
        return new_grid
    return grid

def divide_into_quadrants(grid: ColoredGrid) -> List[ColoredGrid]:
    rows, cols = grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2
    return [
        grid.extract_subgrid(0, 0, mid_row, mid_col),
        grid.extract_subgrid(0, mid_col, mid_row, cols - mid_col),
        grid.extract_subgrid(mid_row, 0, rows - mid_row, mid_col),
        grid.extract_subgrid(mid_row, mid_col, rows - mid_row, cols - mid_col)
    ]

def process_quadrant(quadrant: ColoredGrid) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    patterns = identify_patterns(quadrant)
    return select_pattern(patterns) if patterns else None

def select_pattern(patterns: List[Tuple[int, List[Tuple[int, int]]]]) -> Tuple[int, List[Tuple[int, int]]]:
    return max(patterns, key=lambda x: len(x[1]))

def compress_pattern(pattern: Tuple[int, List[Tuple[int, int]]]) -> List[List[int]]:
    color, cells = pattern
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    
    compressed = [[0 for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            r_start = min_r + (max_r - min_r + 1) * r // 3
            r_end = min_r + (max_r - min_r + 1) * (r + 1) // 3
            c_start = min_c + (max_c - min_c + 1) * c // 3
            c_end = min_c + (max_c - min_c + 1) * (c + 1) // 3
            cells_in_section = [(rr, cc) for rr, cc in cells if r_start <= rr <= r_end and c_start <= cc <= c_end]
            compressed[r][c] = color if cells_in_section else 0
    
    return compressed
