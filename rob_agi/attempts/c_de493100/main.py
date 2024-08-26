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
    3. Segmenting the input grid and identifying dominant colors
    4. Creating an abstracted output grid based on the segmentation
    5. Preserving key patterns and color relationships
    6. Refining the output to balance color distribution and enhance contrast
    
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
    
    # Step 3: Segment Input Grid
    segments = segment_grid(input_grid, output_size)
    
    # Step 4: Create Initial Output Grid
    output_grid = create_initial_grid(segments, output_size)
    
    # Step 5: Preserve Key Patterns
    output_grid = preserve_key_patterns(output_grid, input_grid)
    
    # Step 6: Balance Color Distribution
    output_grid = balance_colors(output_grid, color_freq)
    
    # Step 7: Enhance Contrast
    output_grid = enhance_contrast(output_grid)
    
    return ColoredGrid(values=output_grid)

def analyze_colors(grid: ColoredGrid) -> Dict[int, int]:
    flat_grid = [color for row in grid.values for color in row]
    return dict(Counter(flat_grid))

def analyze_complexity(grid: ColoredGrid) -> int:
    unique_colors = len(set(color for row in grid.values for color in row))
    edge_count = sum(sum(row) for row in detect_edges(grid))
    return unique_colors * 10 + edge_count

def detect_edges(grid: ColoredGrid) -> List[List[bool]]:
    edges = [[False for _ in range(len(grid.values[0]))] for _ in range(len(grid.values))]
    for r in range(len(grid.values)):
        for c in range(len(grid.values[0])):
            if r > 0 and grid.values[r][c] != grid.values[r-1][c]:
                edges[r][c] = True
            if c > 0 and grid.values[r][c] != grid.values[r][c-1]:
                edges[r][c] = True
    return edges

def determine_output_size(complexity: int) -> Tuple[int, int]:
    size = min(max(4, complexity // 20), 10)
    return (size, size)

def segment_grid(grid: ColoredGrid, output_size: Tuple[int, int]) -> List[List[Dict[int, int]]]:
    segment_height = len(grid.values) // output_size[0]
    segment_width = len(grid.values[0]) // output_size[1]
    segments = []
    for r in range(0, len(grid.values), segment_height):
        row = []
        for c in range(0, len(grid.values[0]), segment_width):
            segment = [row[c:c+segment_width] for row in grid.values[r:r+segment_height]]
            color_freq = Counter(color for row in segment for color in row)
            row.append(color_freq)
        segments.append(row)
    return segments

def create_initial_grid(segments: List[List[Dict[int, int]]], output_size: Tuple[int, int]) -> List[List[int]]:
    return [[max(segment, key=segment.get) for segment in row] for row in segments]

def preserve_key_patterns(output_grid: List[List[int]], input_grid: ColoredGrid) -> List[List[int]]:
    # Implement logic to preserve key patterns from input_grid in output_grid
    return output_grid

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

def enhance_contrast(grid: List[List[int]]) -> List[List[int]]:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if r > 0 and grid[r][c] == grid[r-1][c]:
                grid[r][c] = (grid[r][c] + 1) % 10
            if c > 0 and grid[r][c] == grid[r][c-1]:
                grid[r][c] = (grid[r][c] + 1) % 10
    return grid
