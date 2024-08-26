from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import Counter

def solve_0934a4d8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by analyzing the input grid for repeating motifs and constructing a smaller output grid
    that captures the essence of the input's pattern and color distribution.
    
    The function performs the following steps:
    1. Analyzes the input grid for repeating motifs of various sizes
    2. Identifies the most significant motif based on frequency and coverage
    3. Determines the output grid size based on the significant motif
    4. Constructs an output grid that represents the key motif and color distribution
    5. Ensures the output has at least 3 distinct colors and captures the input's essence
    """
    motif, motif_size = find_significant_motif(input_grid)
    output_size = determine_output_size(motif_size)
    output_grid = construct_output_grid(input_grid, motif, output_size)
    
    return output_grid

def find_significant_motif(grid: ColoredGrid) -> Tuple[List[List[int]], Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    max_window_size = min(rows, cols) // 2
    best_motif = None
    best_score = 0
    best_size = (1, 1)

    for window_height in range(2, max_window_size + 1):
        for window_width in range(2, max_window_size + 1):
            motifs = Counter()
            for i in range(0, rows - window_height + 1):
                for j in range(0, cols - window_width + 1):
                    motif = tuple(tuple(grid.values[i+x][j:j+window_width]) for x in range(window_height))
                    motifs[motif] += 1
            
            if motifs:
                most_common_motif, frequency = motifs.most_common(1)[0]
                coverage = frequency * window_height * window_width / (rows * cols)
                score = frequency * coverage
                if score > best_score:
                    best_motif = list(map(list, most_common_motif))
                    best_score = score
                    best_size = (window_height, window_width)

    return best_motif or [[grid.values[0][0]]], best_size

def determine_output_size(motif_size: Tuple[int, int]) -> Tuple[int, int]:
    height, width = motif_size
    if height * width < 9:
        factor = max(2, (9 // (height * width)) + 1)
        return (height * factor, width * factor)
    elif height * width > 81:
        factor = ((height * width) // 81) + 1
        return (height // factor, width // factor)
    else:
        return motif_size

def construct_output_grid(input_grid: ColoredGrid, motif: List[List[int]], size: Tuple[int, int]) -> ColoredGrid:
    rows, cols = size
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Fill the grid with the motif
    motif_height, motif_width = len(motif), len(motif[0])
    for i in range(rows):
        for j in range(cols):
            output[i][j] = motif[i % motif_height][j % motif_width]
    
    # Ensure at least 3 distinct colors
    color_freq = Counter(color for row in input_grid.values for color in row)
    top_colors = [color for color, _ in color_freq.most_common()]
    distinct_colors = set(color for row in output for color in row)
    
    if len(distinct_colors) < 3:
        for i, color in enumerate(top_colors):
            if color not in distinct_colors:
                output[i % rows][i % cols] = color
                distinct_colors.add(color)
            if len(distinct_colors) >= 3:
                break
    
    return ColoredGrid(values=output)
