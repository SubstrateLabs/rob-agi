from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, Counter

def solve_7d18a6fb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 7x7 output grid by identifying and arranging color patterns.
    
    1. Identifies the background color (most common color).
    2. Divides the grid into four quadrants.
    3. For each quadrant:
       a. Identifies distinct color regions using a flood fill algorithm.
       b. Selects the most significant region based on size, color uniqueness, and position.
    4. Compresses each selected region into a 3x3 representation.
    5. Creates a 7x7 output grid:
       a. Places compressed regions in the corners corresponding to their original quadrants.
       b. Fills the central cross with the background color to separate quadrants.
    
    Returns a 7x7 ColoredGrid with the arranged patterns.
    """
    preprocessed_grid = preprocess_grid(input_grid)
    quadrants = divide_into_quadrants(preprocessed_grid)
    patterns = [process_quadrant(quadrant) for quadrant in quadrants]
    compressed_patterns = [compress_pattern(pattern) for pattern in patterns if pattern]
    output_grid = create_output_grid(compressed_patterns)
    return output_grid

def identify_background(grid: ColoredGrid) -> int:
    color_counts = Counter(cell for row in grid.values for cell in row)
    return color_counts.most_common(1)[0][0]

def identify_regions(grid: ColoredGrid, background: int) -> List[Tuple[int, List[Tuple[int, int]]]]:
    regions = []
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
            if (r, c) not in visited and grid.values[r][c] != background:
                color = grid.values[r][c]
                region = dfs(r, c, color)
                if region:
                    regions.append((color, region))

    return regions

def score_region(region: Tuple[int, List[Tuple[int, int]]], quadrant_size: Tuple[int, int], color_rarity: Dict[int, float]) -> float:
    color, cells = region
    size_score = len(cells) / (quadrant_size[0] * quadrant_size[1])
    rarity_score = color_rarity[color]
    
    # Calculate position score (closer to corner is better)
    corner = (0, 0)  # Assuming top-left corner, adjust for other quadrants
    distances = [((r - corner[0])**2 + (c - corner[1])**2)**0.5 for r, c in cells]
    position_score = 1 - (sum(distances) / len(distances)) / ((quadrant_size[0]**2 + quadrant_size[1]**2)**0.5)
    
    return size_score * 0.4 + rarity_score * 0.4 + position_score * 0.2

def select_significant_region(regions: List[Tuple[int, List[Tuple[int, int]]]], quadrant_size: Tuple[int, int], color_rarity: Dict[int, float]) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
    if not regions:
        return None
    scored_regions = [(region, score_region(region, quadrant_size, color_rarity)) for region in regions]
    return max(scored_regions, key=lambda x: x[1])[0]

def compress_region(region: Tuple[int, List[Tuple[int, int]]]) -> List[List[int]]:
    color, cells = region
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

def create_output_grid(compressed_regions: List[Optional[List[List[int]]]], background: int) -> ColoredGrid:
    output = [[background for _ in range(7)] for _ in range(7)]
    quadrants = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    for region, (qr, qc) in zip(compressed_regions, quadrants):
        if region:
            for r in range(3):
                for c in range(3):
                    output[qr + r][qc + c] = region[r][c]
    
    # Fill central cross
    for i in range(7):
        output[3][i] = background
        output[i][3] = background
    
    return ColoredGrid(values=output)

def solve_7d18a6fb(input_grid: ColoredGrid) -> ColoredGrid:
    background = identify_background(input_grid)
    rows, cols = input_grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2
    
    quadrants = [
        input_grid.extract_subgrid(0, 0, mid_row, mid_col),
        input_grid.extract_subgrid(0, mid_col, mid_row, cols - mid_col),
        input_grid.extract_subgrid(mid_row, 0, rows - mid_row, mid_col),
        input_grid.extract_subgrid(mid_row, mid_col, rows - mid_row, cols - mid_col)
    ]
    
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    total_cells = rows * cols
    color_rarity = {color: 1 - (count / total_cells) for color, count in color_counts.items()}
    
    compressed_regions = []
    for quadrant in quadrants:
        regions = identify_regions(quadrant, background)
        significant_region = select_significant_region(regions, quadrant.get_dimensions(), color_rarity)
        if significant_region:
            compressed_regions.append(compress_region(significant_region))
        else:
            compressed_regions.append(None)
    
    return create_output_grid(compressed_regions, background)
