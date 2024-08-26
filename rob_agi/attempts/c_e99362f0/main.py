from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict
import random

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 11x9 grid into a 5x4 output grid by following these steps:
    1. Analyze the top and bottom halves of the input grid separately.
    2. Identify dominant colors and their spatial distribution in each half.
    3. Create a color palette based on the input grid's color distribution.
    4. Generate the output grid using the color palette, maintaining spatial relationships.
    5. Ensure color variety and balance in the output grid.
    6. Make final adjustments for visual appeal and pattern representation.
    """
    
    def analyze_half(grid: List[List[int]], start_row: int, end_row: int) -> Dict[int, int]:
        color_count = defaultdict(int)
        for r in range(start_row, end_row):
            for color in grid[r]:
                if color not in [0, 4]:  # Exclude black and yellow
                    color_count[color] += 1
        return color_count

    top_count = analyze_half(input_grid.values, 0, 5)
    bottom_count = analyze_half(input_grid.values, 6, 11)

    def get_color_palette(top_count: Dict[int, int], bottom_count: Dict[int, int]) -> List[int]:
        all_colors = set(top_count.keys()) | set(bottom_count.keys())
        palette = []
        for color in all_colors:
            weight = top_count.get(color, 0) + bottom_count.get(color, 0)
            palette.extend([color] * weight)
        return palette

    color_palette = get_color_palette(top_count, bottom_count)

    output = [[0 for _ in range(4)] for _ in range(5)]

    # Fill the output grid
    for r in range(5):
        for c in range(4):
            output[r][c] = random.choice(color_palette)

    # Ensure color variety
    all_colors = set(color_palette)
    for color in all_colors:
        if color not in [cell for row in output for cell in row]:
            r, c = random.randint(0, 4), random.randint(0, 3)
            output[r][c] = color

    # Balance color distribution
    color_count = defaultdict(int)
    for row in output:
        for color in row:
            color_count[color] += 1
    
    for r in range(5):
        for c in range(4):
            if color_count[output[r][c]] > 4:
                less_common = min(color_count, key=color_count.get)
                if color_count[less_common] < 2:
                    output[r][c] = less_common
                    color_count[output[r][c]] -= 1
                    color_count[less_common] += 1

    # Maintain spatial relationships
    top_dominant = max(top_count, key=top_count.get)
    bottom_dominant = max(bottom_count, key=bottom_count.get)
    output[0][0] = top_dominant
    output[4][3] = bottom_dominant

    # Final adjustments
    if len(set(output[0])) < 2:
        output[0][1] = random.choice([c for c in all_colors if c != output[0][0]])
    if len(set(output[4])) < 2:
        output[4][2] = random.choice([c for c in all_colors if c != output[4][3]])

    return ColoredGrid(values=output)
