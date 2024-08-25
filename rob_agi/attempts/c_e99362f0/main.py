from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_e99362f0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid into a 5x4 output grid by following these steps:
    1. Analyze the input grid, splitting it into left and right sections.
    2. Create a color importance map based on frequency and clustering.
    3. Initialize a 5x4 output grid and fill it with the most important colors.
    4. Ensure color balance and diversity, with a bias towards sky color.
    5. Create small clusters and maintain spatial relationships from the input.
    6. Make final adjustments to meet specific criteria (color representation, distribution).
    """
    
    def analyze_section(section: List[List[int]]) -> Dict[int, float]:
        color_count = defaultdict(int)
        color_positions = defaultdict(list)
        rows, cols = len(section), len(section[0])
        
        for r in range(rows):
            for c in range(cols):
                color = section[r][c]
                if color != 0:  # Ignore black (empty space)
                    color_count[color] += 1
                    color_positions[color].append((r / rows, c / cols))  # Normalized position
        
        importance = {}
        for color, count in color_count.items():
            avg_pos = [sum(p[i] for p in color_positions[color]) / len(color_positions[color]) for i in range(2)]
            cluster_score = 1 - (avg_pos[0] - 0.5)**2 - (avg_pos[1] - 0.5)**2  # Higher score for central clusters
            importance[color] = count * (1 + cluster_score)
            
            if color == 8:  # Boost importance of sky color
                importance[color] *= 1.2
        
        return importance

    # Analyze input grid
    left_section = [row[:4] for row in input_grid.values]
    right_section = [row[5:] for row in input_grid.values]
    left_importance = analyze_section(left_section)
    right_importance = analyze_section(right_section)
    
    # Initialize output grid
    output = [[0 for _ in range(4)] for _ in range(5)]
    color_count = defaultdict(int)
    
    # Fill output grid
    main_colors = {7, 8, 9, 2}
    for r in range(5):
        for c in range(4):
            importance = left_importance if c < 2 else right_importance
            available_colors = [color for color in main_colors if color_count[color] < 5]
            if not available_colors:
                available_colors = list(main_colors)
            
            color = max(available_colors, key=lambda x: importance.get(x, 0))
            output[r][c] = color
            color_count[color] += 1
            importance[color] *= 0.5  # Reduce importance after using
    
    # Ensure sky color appears in at least 3 rows
    sky_rows = sum(1 for row in output if 8 in row)
    if sky_rows < 3:
        for r in range(5):
            if 8 not in output[r]:
                c = output[r].index(max(output[r], key=lambda x: color_count[x]))
                output[r][c] = 8
                color_count[8] += 1
                color_count[output[r][c]] -= 1
                sky_rows += 1
                if sky_rows == 3:
                    break
    
    # Create at least one 2x2 cluster
    for r in range(4):
        for c in range(3):
            colors = {output[r][c], output[r][c+1], output[r+1][c], output[r+1][c+1]}
            if len(colors) == 1:
                return ColoredGrid(values=output)
    
    # If no 2x2 cluster, create one with the most frequent color
    most_frequent = max(color_count, key=color_count.get)
    output[0][0] = output[0][1] = output[1][0] = output[1][1] = most_frequent
    
    return ColoredGrid(values=output)
