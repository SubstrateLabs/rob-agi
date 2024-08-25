from rob_agi.colored_grid import ColoredGrid
from typing import List, Dict, Tuple
from collections import defaultdict

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 5x4 output grid by following these steps:
    1. Analyze the input grid, counting color frequencies and identifying dominant colors in quadrants.
    2. Initialize the output grid with sky blue (8).
    3. Place main colors (7, 9, 2) based on their relative positions in the input.
    4. Create diagonal patterns and adjust edges to reflect input distribution.
    5. Fine-tune color distribution and add black spaces.
    6. Ensure color balance and create at least one 2x2 cluster.
    7. Verify spatial relationships and make final adjustments.
    """
    
    def analyze_input(grid: List[List[int]]) -> Tuple[Dict[int, int], List[int]]:
        color_count = defaultdict(int)
        quadrants = [defaultdict(int) for _ in range(4)]
        rows, cols = len(grid), len(grid[0])
        mid_row, mid_col = rows // 2, cols // 2
        
        for r in range(rows):
            for c in range(cols):
                color = grid[r][c]
                color_count[color] += 1
                quadrant = (r // mid_row) * 2 + (c // mid_col)
                quadrants[quadrant][color] += 1
        
        dominant_colors = [max(q, key=q.get) for q in quadrants]
        return color_count, dominant_colors

    color_count, dominant_quadrants = analyze_input(input_grid.values)
    main_colors = [7, 9, 2, 8]
    
    # Initialize output grid with sky blue
    output = [[8 for _ in range(4)] for _ in range(5)]
    
    # Place main colors based on input distribution
    for color in [7, 9, 2]:
        placements = 0
        for r in range(5):
            for c in range(4):
                if output[r][c] == 8 and color == dominant_quadrants[(r // 3) * 2 + (c // 2)]:
                    output[r][c] = color
                    placements += 1
                    if placements == 3:
                        break
            if placements == 3:
                break
    
    # Create diagonal patterns
    left_dominant = max(color_count, key=lambda x: color_count[x] if x != 8 else 0)
    right_dominant = max(set(main_colors) - {left_dominant, 8}, key=lambda x: color_count[x])
    for i in range(4):
        if output[i][i] == 8:
            output[i][i] = left_dominant
        if output[i][3-i] == 8:
            output[i][3-i] = right_dominant
    
    # Adjust edges
    for i in range(5):
        if output[i][0] == 8 and color_count[7] > color_count[2]:
            output[i][0] = 7
        if output[i][3] == 8 and color_count[9] > color_count[2]:
            output[i][3] = 9
    
    # Fine-tune distribution
    color_distribution = defaultdict(int)
    for row in output:
        for color in row:
            color_distribution[color] += 1
    
    for color in main_colors:
        while color_distribution[color] < 3:
            for r in range(5):
                for c in range(4):
                    if output[r][c] == 8 and color_distribution[8] > 7:
                        output[r][c] = color
                        color_distribution[color] += 1
                        color_distribution[8] -= 1
                        break
                if color_distribution[color] == 3:
                    break
    
    # Add black spaces
    black_added = 0
    for r in range(5):
        for c in range(4):
            if output[r][c] == output[max(0, r-1)][c] == output[min(4, r+1)][c] == output[r][max(0, c-1)] == output[r][min(3, c+1)]:
                output[r][c] = 0
                black_added += 1
                if black_added == 2:
                    break
        if black_added == 2:
            break
    
    # Ensure at least one 2x2 cluster
    cluster_found = False
    for r in range(4):
        for c in range(3):
            if output[r][c] == output[r][c+1] == output[r+1][c] == output[r+1][c+1] != 0:
                cluster_found = True
                break
        if cluster_found:
            break
    
    if not cluster_found:
        most_common = max(color_distribution, key=color_distribution.get)
        output[0][0] = output[0][1] = output[1][0] = output[1][1] = most_common
    
    return ColoredGrid(values=output)
