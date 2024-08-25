from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input 11x9 grid into a 5x4 output grid by following these steps:
    1. Analyze the input grid, counting color frequencies and identifying dominant colors.
    2. Initialize the output grid with sky blue (8).
    3. Place the most dominant color in a 2x2 square.
    4. Distribute other main colors (7, 9, 2) based on their significance in the input.
    5. Balance the composition with sky blue (8) and add black (0) if necessary.
    6. Create color transitions and reflect the input structure.
    7. Make final adjustments for visual balance and aesthetic appeal.
    """
    
    def analyze_input(grid: List[List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        color_count = defaultdict(int)
        quadrant_colors = defaultdict(int)
        rows, cols = len(grid), len(grid[0])
        mid_row, mid_col = rows // 2, cols // 2
        
        for r in range(rows):
            for c in range(cols):
                color = grid[r][c]
                if color not in [0, 4]:  # Exclude black and yellow
                    color_count[color] += 1
                    quadrant = (r // mid_row) * 2 + (c // mid_col)
                    quadrant_colors[color] |= (1 << quadrant)
        
        return color_count, quadrant_colors

    color_count, quadrant_colors = analyze_input(input_grid.values)
    main_colors = [7, 9, 2, 8]
    
    # Initialize output grid with sky blue
    output = [[8 for _ in range(4)] for _ in range(5)]
    
    # Place the most dominant color in a 2x2 square
    dominant_color = max(color_count, key=color_count.get)
    dominant_quadrant = bin(quadrant_colors[dominant_color]).count('1') - 1
    start_row, start_col = (dominant_quadrant // 2) * 2, (dominant_quadrant % 2) * 2
    for r in range(start_row, start_row + 2):
        for c in range(start_col, start_col + 2):
            output[r][c] = dominant_color
    
    # Distribute other main colors
    for color in main_colors:
        if color != dominant_color and color_count[color] > 0:
            placed = False
            for r in range(5):
                for c in range(4):
                    if output[r][c] == 8:
                        output[r][c] = color
                        placed = True
                        break
                if placed:
                    break
    
    # Balance with sky blue and add black if necessary
    sky_blue_count = sum(row.count(8) for row in output)
    if sky_blue_count < 2:
        for r in range(5):
            if output[r][0] == 8 or output[r][3] == 8:
                continue
            if output[r][0] == output[r][1]:
                output[r][3] = 8
            elif output[r][2] == output[r][3]:
                output[r][0] = 8
    
    if color_count[0] > 0:
        for r in range(5):
            for c in range(4):
                if output[r][c] == 8:
                    output[r][c] = 0
                    break
            if 0 in output[r]:
                break
    
    # Create color transitions and reflect input structure
    for r in range(5):
        if output[r][0] == output[r][1] and output[r][2] == output[r][3] and output[r][0] != output[r][2]:
            output[r][1], output[r][2] = output[r][2], output[r][1]
    
    # Final aesthetic adjustments
    if all(output[0][c] == output[1][c] for c in range(4)):
        output[0][1], output[1][2] = output[1][2], output[0][1]
    
    return ColoredGrid(values=output)
