from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict
import random

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 11x9 grid into a 5x4 output grid by following these steps:
    1. Analyze the input grid by dividing it into four quadrants.
    2. Determine color representation in each quadrant.
    3. Create a color palette based on the most representative colors.
    4. Initialize and fill a 5x4 output grid, maintaining spatial relationships.
    5. Balance color distribution and handle black cells.
    6. Make final adjustments to ensure key spatial characteristics are maintained.
    """
    
    def analyze_quadrant(grid: List[List[int]], start_row: int, end_row: int, start_col: int, end_col: int) -> Dict[int, int]:
        color_count = defaultdict(int)
        total_cells = 0
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                color = grid[r][c]
                if color not in [0, 4]:  # Exclude black and yellow
                    color_count[color] += 1
                    total_cells += 1
        return color_count, total_cells

    # Analyze quadrants
    quadrants = [
        analyze_quadrant(input_grid.values, 0, 5, 0, 4),
        analyze_quadrant(input_grid.values, 0, 5, 5, 9),
        analyze_quadrant(input_grid.values, 6, 11, 0, 4),
        analyze_quadrant(input_grid.values, 6, 11, 5, 9)
    ]

    def get_top_colors(color_count: Dict[int, int], total_cells: int) -> List[int]:
        percentages = {color: count / total_cells for color, count in color_count.items()}
        top_colors = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        return [color for color, percentage in top_colors if percentage >= 0.15][:3]

    # Create color palette
    color_palette = set()
    for quadrant in quadrants:
        color_palette.update(get_top_colors(*quadrant))

    # Initialize output grid
    output = [[0 for _ in range(4)] for _ in range(5)]

    # Fill corner 2x2 areas
    corners = [(0, 0), (0, 2), (3, 0), (3, 2)]
    for i, (r, c) in enumerate(corners):
        top_colors = get_top_colors(*quadrants[i])
        if len(top_colors) >= 2:
            output[r][c] = top_colors[0]
            output[r+1][c+1] = top_colors[1]
            output[r][c+1] = random.choice(top_colors[:2])
            output[r+1][c] = random.choice(top_colors[:2])
        elif len(top_colors) == 1:
            output[r][c] = output[r+1][c+1] = top_colors[0]
            output[r][c+1] = output[r+1][c] = random.choice(list(color_palette))

    # Fill central column
    for r in range(5):
        adjacent_colors = set([output[r][1], output[r][2]])
        available_colors = adjacent_colors.intersection(color_palette)
        if not available_colors:
            available_colors = color_palette
        output[r][2] = random.choice(list(available_colors))

    # Balance color distribution
    color_count = defaultdict(int)
    for row in output:
        for color in row:
            color_count[color] += 1
    
    for r in range(5):
        for c in range(4):
            if color_count[output[r][c]] > 5:
                less_common = min(color_count, key=color_count.get)
                output[r][c] = less_common
                color_count[output[r][c]] -= 1
                color_count[less_common] += 1

    # Handle black cells
    for i, (r, c) in enumerate(corners):
        black_percentage = quadrants[i][0][0] / sum(quadrants[i][0].values()) if sum(quadrants[i][0].values()) > 0 else 0
        if black_percentage > 0.25 and 0 not in [output[r][c], output[r][c+1], output[r+1][c], output[r+1][c+1]]:
            replace_pos = random.choice([(r, c), (r, c+1), (r+1, c), (r+1, c+1)])
            output[replace_pos[0]][replace_pos[1]] = 0

    # Final adjustment
    top_left_dominant = max(quadrants[0][0], key=quadrants[0][0].get)
    bottom_right_dominant = max(quadrants[3][0], key=quadrants[3][0].get)
    if output[0][0] != top_left_dominant:
        output[0][0], output[4][3] = top_left_dominant, output[0][0]
    if output[4][3] != bottom_right_dominant:
        output[4][3], output[0][0] = bottom_right_dominant, output[4][3]

    return ColoredGrid(values=output)
