from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_0934a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by analyzing the input grid and constructing a smaller output grid
    that captures the essence of the input's color distribution and transitions.
    
    The function performs the following steps:
    1. Analyzes the input grid for color frequencies and transitions
    2. Determines an appropriate output size based on the input's complexity
    3. Constructs an output grid that represents key colors and transitions
    4. Ensures the output has at least 3 distinct colors and captures the input's essence
    """
    rows, cols = input_grid.get_dimensions()
    color_freq = analyze_color_frequency(input_grid)
    transitions = analyze_color_transitions(input_grid)
    
    output_size = determine_output_size(color_freq, transitions)
    output_grid = construct_output_grid(color_freq, transitions, output_size)
    
    return output_grid

def analyze_color_frequency(grid: ColoredGrid) -> Dict[int, int]:
    return Counter(color for row in grid.values for color in row)

def analyze_color_transitions(grid: ColoredGrid) -> Dict[Tuple[int, int], int]:
    transitions = Counter()
    for row in grid.values:
        for i in range(len(row) - 1):
            transitions[(row[i], row[i+1])] += 1
    return transitions

def determine_output_size(color_freq: Dict[int, int], transitions: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
    unique_colors = len(color_freq)
    complexity = len(transitions)
    if unique_colors <= 4 and complexity <= 10:
        return (3, 3)
    elif unique_colors <= 6 and complexity <= 20:
        return (4, 4)
    else:
        return (5, 5)

def construct_output_grid(color_freq: Dict[int, int], transitions: Dict[Tuple[int, int], int], size: Tuple[int, int]) -> ColoredGrid:
    rows, cols = size
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Fill the grid with the most frequent colors
    top_colors = [color for color, _ in sorted(color_freq.items(), key=lambda x: x[1], reverse=True)]
    for i in range(rows):
        for j in range(cols):
            output[i][j] = top_colors[(i * cols + j) % len(top_colors)]
    
    # Ensure at least 3 distinct colors
    if len(set(top_colors[:rows*cols])) < 3:
        for i in range(3):
            output[i][i] = top_colors[i]
    
    # Apply some common transitions
    top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for (color1, color2), _ in top_transitions[:min(5, len(top_transitions))]:
        for i in range(rows):
            for j in range(cols - 1):
                if output[i][j] == color1:
                    output[i][j+1] = color2
                    break
            if j < cols - 1:
                break
    
    return ColoredGrid(values=output)
