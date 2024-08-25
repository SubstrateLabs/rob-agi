from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 11x9 grid into a 5x4 output grid by following these steps:
    1. Analyze the input grid, counting color frequencies and identifying dominant colors.
    2. Initialize the output grid.
    3. Place the most dominant color in a 2x2 square.
    4. Distribute other main colors based on their significance in the input.
    5. Balance the composition and create color transitions.
    6. Make final adjustments for visual balance and aesthetic appeal.
    """
    
    def analyze_input(grid: List[List[int]]) -> Dict[int, int]:
        color_count = defaultdict(int)
        for row in grid:
            for color in row:
                if color not in [0, 4]:  # Exclude black and yellow
                    color_count[color] += 1
        return color_count

    color_count = analyze_input(input_grid.values)
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)
    
    # Initialize output grid
    output = [[8 for _ in range(4)] for _ in range(5)]
    
    # Place the most dominant color in a 2x2 square
    dominant_color = sorted_colors[0][0]
    for r in range(2):
        for c in range(2):
            output[r][c] = dominant_color
    
    # Distribute other main colors
    color_index = 1
    for r in range(5):
        for c in range(4):
            if output[r][c] == 8 and color_index < len(sorted_colors):
                output[r][c] = sorted_colors[color_index][0]
                color_index += 1
    
    # Balance composition and create transitions
    for r in range(5):
        if output[r][0] == output[r][1] and output[r][2] == output[r][3] and output[r][0] != output[r][2]:
            output[r][1], output[r][2] = output[r][2], output[r][1]
    
    # Final aesthetic adjustments
    if output[2] == output[3]:
        output[2][1], output[3][2] = output[3][2], output[2][1]
    
    return ColoredGrid(values=output)
