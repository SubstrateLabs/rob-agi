from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Dict, Tuple

def solve_3d31c5b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 12x6 input grid into a 3x6 output grid by analyzing and condensing color patterns.
    
    The function divides the input into three 4x6 sections and analyzes them to identify
    significant colors and patterns. It then generates an output grid that preserves key
    characteristics of each section while maintaining some vertical continuity.
    
    Key steps:
    1. Analyze each section of the input grid to determine color significance
    2. Identify dominant colors and patterns for each section
    3. Initialize the output grid with significant edge colors
    4. Fill the output grid based on color significance and patterns
    5. Ensure vertical continuity where appropriate
    6. Balance color representation and make final adjustments
    7. Validate and return the final output grid
    
    Args:
    input_grid (ColoredGrid): A 12x6 grid representing the input pattern
    
    Returns:
    ColoredGrid: A 3x6 grid representing the transformed output pattern
    """
    rows, cols = input_grid.get_dimensions()
    section_height = rows // 3

    def analyze_section(section: List[List[int]]) -> Dict[int, float]:
        counter = Counter()
        for r, row in enumerate(section):
            for c, color in enumerate(row):
                if color != 0:
                    # Weight by position (center and edges are more significant)
                    weight = 1.5 if c in (0, len(row)-1) or r in (0, len(section)-1) else 1.0
                    counter[color] += weight
        return dict(counter)

    def get_vertical_pattern(grid: List[List[int]], col: int) -> List[int]:
        return [row[col] for row in grid if row[col] != 0]

    def get_significant_colors(analysis: Dict[int, float], n: int = 3) -> List[int]:
        return sorted(analysis, key=analysis.get, reverse=True)[:n]

    # Analyze sections
    section_analyses = [analyze_section(input_grid.values[i*section_height:(i+1)*section_height]) for i in range(3)]

    # Initialize output grid
    output_values = [[0 for _ in range(cols)] for _ in range(3)]

    # Set edge colors
    for i in range(3):
        significant_colors = get_significant_colors(section_analyses[i])
        left_edge = input_grid.values[i*section_height][0]
        right_edge = input_grid.values[i*section_height + section_height - 1][-1]
        output_values[i][0] = left_edge if left_edge in significant_colors else significant_colors[0]
        output_values[i][-1] = right_edge if right_edge in significant_colors else significant_colors[-1]

    # Fill output grid
    for i in range(3):
        significant_colors = get_significant_colors(section_analyses[i], 4)
        for j in range(1, cols-1):
            vertical_pattern = get_vertical_pattern(input_grid.values, j)
            color_scores = {color: section_analyses[i].get(color, 0) for color in significant_colors}
            
            # Consider vertical continuity
            for color in vertical_pattern:
                if color in color_scores:
                    color_scores[color] += vertical_pattern.count(color) * 0.5
            
            # Choose color
            chosen_color = max(color_scores, key=color_scores.get)
            output_values[i][j] = chosen_color

    # Ensure vertical continuity
    for j in range(cols):
        column = [output_values[i][j] for i in range(3)]
        if column[0] == column[2] and column[0] != column[1]:
            if column[0] in get_significant_colors(section_analyses[1]):
                output_values[1][j] = column[0]

    # Balance color representation
    for i in range(3):
        row_colors = set(output_values[i])
        for color in get_significant_colors(section_analyses[i], 3):
            if color not in row_colors:
                least_significant = min(range(1, cols-1), key=lambda x: section_analyses[i].get(output_values[i][x], 0))
                output_values[i][least_significant] = color

    # Final adjustments
    for i in range(3):
        if len(set(output_values[i])) < 2:
            output_values[i][cols//2] = get_significant_colors(section_analyses[i], 3)[-1]

    return ColoredGrid(values=output_values)
