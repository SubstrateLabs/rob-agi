from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict
import random

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 11x9 grid into a 5x4 output grid by following these steps:
    1. Analyze the top and bottom halves of the input grid separately.
    2. Identify dominant colors and their spatial distribution in each half.
    3. Initialize the output grid with a neutral color.
    4. Place colors in the output grid based on their significance and spatial relationships in the input.
    5. Preserve strong color patterns and vertical relationships from the input.
    6. Balance color distribution and create smooth transitions.
    7. Make final adjustments for visual appeal and pattern representation.
    """
    
    def analyze_half(grid: List[List[int]], start_row: int, end_row: int) -> Tuple[Dict[int, int], Dict[int, List[Tuple[int, int]]]]:
        color_count = defaultdict(int)
        color_positions = defaultdict(list)
        for r in range(start_row, end_row):
            for c, color in enumerate(grid[r]):
                if color not in [0, 4]:  # Exclude black and yellow
                    color_count[color] += 1
                    color_positions[color].append((r - start_row, c))
        return color_count, color_positions

    top_count, top_positions = analyze_half(input_grid.values, 0, 5)
    bottom_count, bottom_positions = analyze_half(input_grid.values, 6, 11)

    def get_main_colors(count_dict: Dict[int, int], n: int) -> List[int]:
        return [color for color, _ in sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:n]]

    top_colors = get_main_colors(top_count, 3)
    bottom_colors = get_main_colors(bottom_count, 3)
    main_colors = list(set(top_colors + bottom_colors))

    output = [[8 for _ in range(4)] for _ in range(5)]

    def place_color(color: int, half: str):
        positions = top_positions if half == 'top' else bottom_positions
        for _ in range(2):  # Try to place the color twice
            if not positions[color]:
                return
            r, c = random.choice(positions[color])
            output_r = r if half == 'top' else r + 2
            output_c = c // 3 if half == 'top' else 2 + (c // 3)
            if 0 <= output_r < 5 and 0 <= output_c < 4:
                output[output_r][output_c] = color

    for color in main_colors:
        if color in top_colors:
            place_color(color, 'top')
        if color in bottom_colors:
            place_color(color, 'bottom')

    # Ensure all main colors are represented
    for color in main_colors:
        if color not in [cell for row in output for cell in row]:
            r, c = random.randint(0, 4), random.randint(0, 3)
            output[r][c] = color

    # Create transitions and balance
    for r in range(5):
        if output[r][0] == output[r][1] and output[r][2] == output[r][3] and output[r][0] != output[r][2]:
            output[r][1], output[r][2] = output[r][2], output[r][1]

    # Final adjustments
    if output[2] == output[3]:
        output[2][1], output[3][2] = output[3][2], output[2][1]

    return ColoredGrid(values=output)
