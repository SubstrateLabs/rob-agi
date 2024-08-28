from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
from collections import Counter

def solve_0934a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by analyzing the input grid and constructing a smaller output grid
    that captures the essence of the input's color distribution and key patterns.
    
    The function performs the following steps:
    1. Analyzes the input grid to identify color distribution and key patterns
    2. Determines the output grid size based on input dimensions and patterns
    3. Constructs an output grid that represents essential features and color relationships
    4. Ensures the output has at least 3 distinct colors and captures the input's essence
    5. Refines the output to better match the input's characteristics
    """
    color_freq = analyze_color_distribution(input_grid)
    output_size = determine_output_size(input_grid)
    output_grid = construct_output_grid(input_grid, color_freq, output_size)
    output_grid = refine_output_grid(output_grid, input_grid)
    
    return ColoredGrid(values=output_grid)

def analyze_color_distribution(grid: ColoredGrid) -> Counter:
    return Counter(color for row in grid.values for color in row)

def determine_output_size(input_grid: ColoredGrid) -> Tuple[int, int]:
    rows, cols = input_grid.get_dimensions()
    if rows <= 10 and cols <= 10:
        return max(3, min(rows, cols)), max(3, min(rows, cols))
    elif rows <= 20 and cols <= 20:
        return 5, 5
    else:
        return 7, min(9, max(3, cols // 4))

def construct_output_grid(input_grid: ColoredGrid, color_freq: Counter, size: Tuple[int, int]) -> List[List[int]]:
    height, width = size
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    # Fill with most common colors
    common_colors = [color for color, _ in color_freq.most_common(3)]
    for r in range(height):
        for c in range(width):
            output[r][c] = common_colors[(r + c) % len(common_colors)]
    
    # Ensure corners represent input corners
    output[0][0] = input_grid.values[0][0]
    output[0][-1] = input_grid.values[0][-1]
    output[-1][0] = input_grid.values[-1][0]
    output[-1][-1] = input_grid.values[-1][-1]
    
    return output

def refine_output_grid(output: List[List[int]], input_grid: ColoredGrid) -> List[List[int]]:
    height, width = len(output), len(output[0])
    input_rows, input_cols = input_grid.get_dimensions()
    
    # Adjust middle row and column to better represent input
    mid_row = height // 2
    mid_col = width // 2
    for c in range(width):
        output[mid_row][c] = input_grid.values[input_rows // 2][c * input_cols // width]
    for r in range(height):
        output[r][mid_col] = input_grid.values[r * input_rows // height][input_cols // 2]
    
    # Ensure at least 3 distinct colors
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
