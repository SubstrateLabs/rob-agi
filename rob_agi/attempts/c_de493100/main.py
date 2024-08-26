from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
import numpy as np
from collections import Counter

def solve_de493100(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the de493100 challenge by creating an abstracted version of the input grid.
    
    The solution works by:
    1. Analyzing color frequencies and patterns in the input grid
    2. Creating a color mapping to simplify the color palette
    3. Detecting edges and significant color transitions
    4. Abstracting the grid into regions based on color similarity
    5. Generating a smaller output grid that captures the essence of the input
    
    The algorithm aims to balance:
    - Color representation (maintaining dominant colors and their relationships)
    - Pattern preservation (capturing significant recurring patterns)
    - Structural abstraction (representing the overall layout in a simplified form)
    
    Args:
    input_grid (ColoredGrid): The input 30x30 grid

    Returns:
    ColoredGrid: An abstracted version of the input grid, sized between 4x4 and 10x10
    """
    # Step 1: Color Analysis
    color_freq = analyze_colors(input_grid)
    color_mapping = create_color_mapping(color_freq)
    
    # Step 2: Pattern Recognition
    patterns = recognize_patterns(input_grid, color_mapping)
    
    # Step 3: Edge Detection
    edges = detect_edges(input_grid, color_mapping)
    
    # Step 4: Grid Abstraction
    regions = abstract_grid(input_grid, color_mapping, edges)
    
    # Step 5: Output Grid Size Determination
    output_size = determine_output_size(regions, patterns)
    
    # Step 6: Abstract Grid Generation
    output_grid = generate_abstract_grid(regions, patterns, output_size)
    
    # Step 7: Pattern Reinforcement
    output_grid = reinforce_patterns(output_grid, patterns)
    
    # Step 8: Color Refinement
    output_grid = refine_colors(output_grid, color_freq)
    
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

def generate_abstract_grid(regions: List[List[Tuple[int, int]]], patterns: List[Tuple[List[int], int]], size: Tuple[int, int]) -> List[List[int]]:
    output = [[0 for _ in range(size[1])] for _ in range(size[0])]
    for region in regions:
        color = sum(r + c for r, c in region) % 10  # Simple way to assign a color based on region
        center_r = sum(r for r, _ in region) // len(region)
        center_c = sum(c for _, c in region) // len(region)
        output_r = center_r * size[0] // 30
        output_c = center_c * size[1] // 30
        output[output_r][output_c] = color
    return output

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
