from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict
import random

from typing import List, Tuple, Dict
from collections import Counter

def solve_f4081712(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f4081712 challenge by analyzing the input grid and generating a condensed
    representation that captures key color relationships and patterns.
    
    The solution involves the following steps:
    1. Analyze the entire input grid, focusing on the central area and key patterns.
    2. Identify the most significant colors and their relationships.
    3. Determine the appropriate output size based on the complexity of the input.
    4. Generate a smaller output grid that preserves the essence of the input pattern.
    5. Ensure the output maintains key color relationships and relative positions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: A condensed representation of the input grid, capturing its essential pattern.
    """
    analysis = analyze_grid(input_grid)
    key_colors = identify_key_colors(analysis)
    output_size = determine_output_size(analysis)
    output_values = generate_output_grid(key_colors, output_size, analysis)
    return ColoredGrid(values=output_values)

def analyze_grid(grid: ColoredGrid) -> Dict:
    rows, cols = len(grid.values), len(grid.values[0])
    central_area = [row[cols//4:3*cols//4] for row in grid.values[rows//4:3*rows//4]]
    
    analysis = {
        'full_freq': Counter(cell for row in grid.values for cell in row),
        'central_freq': Counter(cell for row in central_area for cell in row),
        'transitions': Counter(),
        'edge_colors': set(grid.values[0] + grid.values[-1] + [row[0] for row in grid.values] + [row[-1] for row in grid.values])
    }
    
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i][j+1])] += 1
            if i < rows - 1:
                analysis['transitions'][(grid.values[i][j], grid.values[i+1][j])] += 1
    
    return analysis

def identify_key_colors(analysis: Dict) -> List[int]:
    combined_freq = analysis['central_freq'] + Counter({color: count // 2 for color, count in analysis['full_freq'].items()})
    return [color for color, _ in combined_freq.most_common(5)]  # Reduced to top 5 colors

def determine_output_size(analysis: Dict) -> Tuple[int, int]:
    unique_colors = len(analysis['central_freq'])
    size = max(3, min(5, unique_colors + 1))  # Adjusted to produce smaller outputs
    return (size, size)

def generate_output_grid(key_colors: List[int], output_size: Tuple[int, int], analysis: Dict) -> List[List[int]]:
    rows, cols = output_size
    output = [[0] * cols for _ in range(rows)]
    
    # Place key colors
    for i, color in enumerate(key_colors):
        r, c = i % rows, i % cols
        output[r][c] = color
    
    # Fill remaining cells based on transitions
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 0:
                neighbors = [output[r-1][c] if r > 0 else None,
                             output[r][c-1] if c > 0 else None]
                neighbors = [n for n in neighbors if n is not None]
                if neighbors:
                    possible_colors = [color2 for (color1, color2), _ in analysis['transitions'].most_common()
                                       if color1 in neighbors]
                    if possible_colors:
                        output[r][c] = possible_colors[0]
                    else:
                        output[r][c] = key_colors[0]
                else:
                    output[r][c] = key_colors[0]
    
    return output
