from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_0934a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by analyzing the input grid for significant patterns and constructing a smaller output grid
    that captures the essence of the input's color distribution and arrangement.
    
    The function performs the following steps:
    1. Analyzes the input grid to identify regions of interest and color distribution
    2. Extracts significant patterns using a sliding window approach
    3. Determines the output grid size based on the most significant patterns
    4. Constructs an output grid that represents key patterns and color distribution
    5. Ensures the output has at least 3 distinct colors and captures the input's essence
    """
    patterns = extract_significant_patterns(input_grid)
    output_size = determine_output_size(patterns, input_grid)
    output_grid = construct_output_grid(input_grid, patterns, output_size)
    
    return ColoredGrid(values=output_grid)

def extract_significant_patterns(grid: ColoredGrid) -> List[Tuple[List[List[int]], int]]:
    rows, cols = grid.get_dimensions()
    patterns = []
    
    for size in range(3, min(10, min(rows, cols) + 1)):
        for r in range(rows - size + 1):
            for c in range(cols - size + 1):
                pattern = [row[c:c+size] for row in grid.values[r:r+size]]
                score = calculate_pattern_score(pattern, grid)
                patterns.append((pattern, score))
    
    patterns.sort(key=lambda x: x[1], reverse=True)
    return patterns[:5]  # Return top 5 patterns

def calculate_pattern_score(pattern: List[List[int]], grid: ColoredGrid) -> int:
    color_diversity = len(set(color for row in pattern for color in row))
    pattern_size = len(pattern)
    color_representation = sum(grid.count_color(color) for row in pattern for color in row)
    return color_diversity * pattern_size * color_representation

def determine_output_size(patterns: List[Tuple[List[List[int]], int]], input_grid: ColoredGrid) -> Tuple[int, int]:
    input_rows, input_cols = input_grid.get_dimensions()
    best_pattern_size = len(patterns[0][0])
    
    if input_rows <= 10 and input_cols <= 10:
        return max(3, min(input_rows, input_cols)), max(3, min(input_rows, input_cols))
    elif input_rows <= 20 and input_cols <= 20:
        return best_pattern_size, best_pattern_size
    else:
        return min(9, max(4, best_pattern_size)), min(9, max(4, best_pattern_size))

def construct_output_grid(input_grid: ColoredGrid, patterns: List[Tuple[List[List[int]], int]], size: Tuple[int, int]) -> List[List[int]]:
    height, width = size
    best_pattern = patterns[0][0]
    pattern_size = len(best_pattern)
    
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for r in range(height):
        for c in range(width):
            output[r][c] = best_pattern[r % pattern_size][c % pattern_size]
    
    # Ensure color diversity
    distinct_colors = set(color for row in output for color in row)
    if len(distinct_colors) < 3:
        color_freq = Counter(color for row in input_grid.values for color in row)
        for color, _ in color_freq.most_common():
            if color not in distinct_colors:
                output[len(distinct_colors) % height][0] = color
                distinct_colors.add(color)
            if len(distinct_colors) >= 3:
                break
    
    return output
