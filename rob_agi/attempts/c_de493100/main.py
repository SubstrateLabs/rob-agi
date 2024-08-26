from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter

def solve_de493100(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the de493100 challenge by creating an abstracted version of the input grid.
    
    The solution works by:
    1. Analyzing input complexity and color frequencies
    2. Determining appropriate output size
    3. Identifying key structural elements and patterns
    4. Generating a low-resolution grid that captures essential features
    5. Refining the output to balance color distribution and enhance contrast
    
    The algorithm aims to balance:
    - Color representation (maintaining dominant colors and their relationships)
    - Pattern preservation (capturing significant recurring patterns)
    - Structural abstraction (representing the overall layout in a simplified form)
    
    Args:
    input_grid (ColoredGrid): The input 30x30 grid

    Returns:
    ColoredGrid: An abstracted version of the input grid, sized between 4x4 and 10x10
    """
    # Step 1: Analyze Input Complexity
    color_freq = analyze_colors(input_grid)
    complexity = analyze_complexity(input_grid)
    
    # Step 2: Determine Output Size
    output_size = determine_output_size(complexity)
    
    # Step 3: Create Color Importance Map
    color_importance = create_color_importance_map(color_freq, input_grid)
    
    # Step 4: Identify Key Structural Elements
    edges = detect_edges(input_grid)
    key_regions = identify_key_regions(input_grid, color_importance)
    
    # Step 5: Generate Low-Resolution Grid
    output_grid = generate_low_res_grid(output_size, color_importance)
    
    # Step 6: Place Key Structural Elements
    output_grid = place_key_elements(output_grid, key_regions, edges, output_size)
    
    # Step 7: Fill in Details
    output_grid = fill_details(output_grid, color_importance)
    
    # Step 8: Enhance Contrast
    output_grid = enhance_contrast(output_grid)
    
    # Step 9: Refine Patterns
    output_grid = refine_patterns(output_grid, input_grid)
    
    # Step 10: Balance Color Distribution
    output_grid = balance_colors(output_grid, color_freq)
    
    return ColoredGrid(values=output_grid)

def analyze_colors(grid: ColoredGrid) -> Dict[int, int]:
    flat_grid = [color for row in grid.values for color in row]
    return dict(Counter(flat_grid))

def create_color_mapping(color_freq: Dict[int, int]) -> Dict[int, int]:
    sorted_colors = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)
    top_colors = [color for color, _ in sorted_colors[:4]]
    return {color: (color if color in top_colors else top_colors[0]) for color in color_freq}

def recognize_patterns(grid: ColoredGrid, color_mapping: Dict[int, int]) -> List[Tuple[List[int], int]]:
    patterns = []
    for row in grid.values:
        mapped_row = [color_mapping[color] for color in row]
        for i in range(len(mapped_row) - 2):
            pattern = tuple(mapped_row[i:i+3])
            patterns.append(pattern)
    return Counter(patterns).most_common(5)

def detect_edges(grid: ColoredGrid, color_mapping: Dict[int, int]) -> List[List[bool]]:
    edges = [[False for _ in range(len(grid.values[0]))] for _ in range(len(grid.values))]
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if r > 0 and color_mapping[grid.values[r][c]] != color_mapping[grid.values[r-1][c]]:
                edges[r][c] = True
            if c > 0 and color_mapping[grid.values[r][c]] != color_mapping[grid.values[r][c-1]]:
                edges[r][c] = True
    return edges

def abstract_grid(grid: ColoredGrid, color_mapping: Dict[int, int], edges: List[List[bool]]) -> List[List[int]]:
    regions = []
    visited = set()
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if (r, c) not in visited:
                region = flood_fill(grid, color_mapping, edges, r, c, visited)
                regions.append(region)
    return regions

def flood_fill(grid: ColoredGrid, color_mapping: Dict[int, int], edges: List[List[bool]], r: int, c: int, visited: set) -> List[Tuple[int, int]]:
    queue = [(r, c)]
    region = []
    target_color = color_mapping[grid.values[r][c]]
    while queue:
        r, c = queue.pop(0)
        if (r, c) in visited or edges[r][c]:
            continue
        visited.add((r, c))
        region.append((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid.values) and 0 <= nc < len(grid.values[0]) and color_mapping[grid.values[nr][nc]] == target_color:
                queue.append((nr, nc))
    return region

def determine_output_size(regions: List[List[Tuple[int, int]]], patterns: List[Tuple[List[int], int]]) -> Tuple[int, int]:
    complexity = len(regions) + len(patterns)
    size = min(max(4, complexity // 2), 10)
    return (size, size)

def analyze_complexity(grid: ColoredGrid) -> int:
    unique_colors = len(set(color for row in grid.values for color in row))
    edge_count = sum(sum(row) for row in detect_edges(grid))
    return unique_colors * 10 + edge_count

def determine_output_size(complexity: int) -> Tuple[int, int]:
    size = min(max(4, complexity // 20), 10)
    return (size, size)

def create_color_importance_map(color_freq: Dict[int, int], grid: ColoredGrid) -> List[Tuple[int, float]]:
    total_cells = sum(color_freq.values())
    importance = [(color, count / total_cells) for color, count in color_freq.items()]
    return sorted(importance, key=lambda x: x[1], reverse=True)

def identify_key_regions(grid: ColoredGrid, color_importance: List[Tuple[int, float]]) -> List[Tuple[int, int, int, int]]:
    key_regions = []
    for color, _ in color_importance[:3]:  # Consider top 3 colors
        regions = grid.find_connected_regions(color)
        if regions:
            largest_region = max(regions, key=len)
            min_r = min(r for r, _ in largest_region)
            max_r = max(r for r, _ in largest_region)
            min_c = min(c for _, c in largest_region)
            max_c = max(c for _, c in largest_region)
            key_regions.append((min_r, min_c, max_r, max_c))
    return key_regions

def generate_low_res_grid(size: Tuple[int, int], color_importance: List[Tuple[int, float]]) -> List[List[int]]:
    dominant_color = color_importance[0][0]
    return [[dominant_color for _ in range(size[1])] for _ in range(size[0])]

def place_key_elements(grid: List[List[int]], key_regions: List[Tuple[int, int, int, int]], edges: List[List[bool]], output_size: Tuple[int, int]) -> List[List[int]]:
    for i, (min_r, min_c, max_r, max_c) in enumerate(key_regions):
        color = i + 1  # Use different colors for each key region
        r_scale = output_size[0] / 30
        c_scale = output_size[1] / 30
        for r in range(int(min_r * r_scale), int(max_r * r_scale) + 1):
            for c in range(int(min_c * c_scale), int(max_c * c_scale) + 1):
                if 0 <= r < output_size[0] and 0 <= c < output_size[1]:
                    grid[r][c] = color
    return grid

def fill_details(grid: List[List[int]], color_importance: List[Tuple[int, float]]) -> List[List[int]]:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0:  # Fill in empty spaces
                grid[r][c] = color_importance[r % len(color_importance)][0]
    return grid

def enhance_contrast(grid: List[List[int]]) -> List[List[int]]:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if r > 0 and grid[r][c] == grid[r-1][c]:
                grid[r][c] = (grid[r][c] + 1) % 10
            if c > 0 and grid[r][c] == grid[r][c-1]:
                grid[r][c] = (grid[r][c] + 1) % 10
    return grid

def refine_patterns(grid: List[List[int]], input_grid: ColoredGrid) -> List[List[int]]:
    patterns = recognize_patterns(input_grid, {})
    for r in range(len(grid) - 2):
        for c in range(len(grid[0]) - 2):
            subgrid = [grid[r+i][c:c+3] for i in range(3)]
            if tuple(subgrid[0] + subgrid[1] + subgrid[2]) in patterns:
                for i in range(3):
                    for j in range(3):
                        grid[r+i][c+j] = patterns[0][0][i*3+j]
    return grid

def balance_colors(grid: List[List[int]], color_freq: Dict[int, int]) -> List[List[int]]:
    target_freq = {color: count / sum(color_freq.values()) for color, count in color_freq.items()}
    current_freq = Counter(color for row in grid for color in row)
    current_freq = {color: count / sum(current_freq.values()) for color, count in current_freq.items()}
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            current_color = grid[r][c]
            if current_freq[current_color] > target_freq.get(current_color, 0):
                new_color = min(color_freq.keys(), key=lambda x: current_freq.get(x, 0) - target_freq.get(x, 0))
                grid[r][c] = new_color
                current_freq[current_color] -= 1 / (len(grid) * len(grid[0]))
                current_freq[new_color] = current_freq.get(new_color, 0) + 1 / (len(grid) * len(grid[0]))
    
    return grid

def reinforce_patterns(grid: List[List[int]], patterns: List[Tuple[List[int], int]]) -> List[List[int]]:
    for r in range(len(grid)):
        for c in range(len(grid[0]) - 2):
            if tuple(grid[r][c:c+3]) not in patterns:
                grid[r][c:c+3] = patterns[0][0]
    return grid

def refine_colors(grid: List[List[int]], color_freq: Dict[int, int]) -> List[List[int]]:
    top_colors = sorted(color_freq.items(), key=lambda x: x[1], reverse=True)[:4]
    color_map = {i: color for i, (color, _) in enumerate(top_colors)}
    return [[color_map.get(cell, cell) for cell in row] for row in grid]
