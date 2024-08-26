from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict
import random

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 11x9 grid into a 5x4 output grid by following these steps:
    1. Analyze the top and bottom halves of the input grid separately.
    2. Identify dominant colors and their spatial distribution in each half.
    3. Initialize the output grid.
    4. Place dominant colors from the input into corresponding areas of the output.
    5. Create transitions and balance color distribution.
    6. Maintain spatial relationships and introduce variety.
    7. Make final adjustments for visual appeal and pattern representation.
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

    def get_main_colors(count_dict: Dict[int, int], n: int) -> List[int]:
        return [color for color, _ in sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:n]]

    top_colors = get_main_colors(top_count, 3)
    bottom_colors = get_main_colors(bottom_count, 3)
    all_colors = list(set(top_colors + bottom_colors))

    output = [[0 for _ in range(4)] for _ in range(5)]

    # Place top colors
    for r in range(2):
        for c in range(4):
            output[r][c] = random.choice(top_colors)

    # Place bottom colors
    for r in range(3, 5):
        for c in range(4):
            output[r][c] = random.choice(bottom_colors)

    # Create transitions in the middle row
    output[2] = [random.choice(all_colors) for _ in range(4)]

    # Ensure all main colors are represented
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
            if color_count[output[r][c]] > 3:
                less_common = min(color_count, key=color_count.get)
                if color_count[less_common] < 2:
                    output[r][c] = less_common
                    color_count[output[r][c]] -= 1
                    color_count[less_common] += 1

    # Final adjustments
    if output[0] == output[1]:
        output[1][1], output[1][2] = output[1][2], output[1][1]
    if output[3] == output[4]:
        output[4][1], output[4][2] = output[4][2], output[4][1]

    return ColoredGrid(values=output)
