from rob_agi.colored_grid import ColoredGrid
from collections import Counter

from collections import Counter
from typing import List, Tuple

def solve_3d31c5b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 12x6 input grid into a 3x6 output grid by analyzing color patterns, flows, and structures.
    
    The function analyzes the entire input grid for color frequency, continuity, edge patterns, and diagonal flows.
    It divides the input into three 4x6 sections and creates a condensed representation in the output.
    The algorithm preserves dominant colors, significant patterns, structural elements, and overall color distribution
    while reducing the vertical dimension.
    
    Steps:
    1. Analyze the entire input grid for color frequency, patterns, and flows
    2. Divide the input grid into three sections and analyze each
    3. Create a color scoring system based on frequency, continuity, edge patterns, and diagonal flows
    4. Generate the output grid based on color scores, section relevance, and structural importance
    5. Post-process to ensure balanced representation, pattern preservation, and structural integrity
    
    Args:
    input_grid (ColoredGrid): A 12x6 grid representing the input pattern
    
    Returns:
    ColoredGrid: A 3x6 grid representing the transformed output pattern
    """
    rows, cols = input_grid.get_dimensions()
    section_height = rows // 3

    def analyze_grid(grid: List[List[int]]) -> Counter:
        return Counter(cell for row in grid for cell in row if cell != 0)

    def analyze_continuity(grid: List[List[int]]) -> dict:
        continuity_scores = {}
        for direction in ['vertical', 'horizontal', 'diagonal']:
            for i in range(rows if direction != 'horizontal' else cols):
                if direction == 'vertical':
                    line = [grid[row][i] for row in range(rows)]
                elif direction == 'horizontal':
                    line = grid[i]
                else:  # diagonal
                    line = [grid[row][(i + row) % cols] for row in range(rows)]
                
                streak = 1
                for j in range(1, len(line)):
                    if line[j] == line[j-1] and line[j] != 0:
                        streak += 1
                    else:
                        if line[j-1] != 0:
                            continuity_scores[line[j-1]] = max(continuity_scores.get(line[j-1], 0), streak)
                        streak = 1
                if line[-1] != 0:
                    continuity_scores[line[-1]] = max(continuity_scores.get(line[-1], 0), streak)
        return continuity_scores

    def analyze_edge_patterns(grid: List[List[int]]) -> dict:
        edges = Counter(row[0] for row in grid if row[0] != 0) + \
                Counter(row[-1] for row in grid if row[-1] != 0) + \
                Counter(grid[0]) + Counter(grid[-1])
        return dict(edges)

    def analyze_structural_role(grid: List[List[int]]) -> dict:
        structural_scores = {}
        for color in set(cell for row in grid for cell in row):
            if color == 0:  # Skip black (empty space)
                continue
            regions = input_grid.find_connected_regions(color)
            structural_scores[color] = sum(len(region) for region in regions)
        return structural_scores

    overall_counter = analyze_grid(input_grid.values)
    continuity_scores = analyze_continuity(input_grid.values)
    edge_scores = analyze_edge_patterns(input_grid.values)
    structural_scores = analyze_structural_role(input_grid.values)

    def color_score(color: int, section_counter: Counter, row: int, col: int) -> float:
        base_score = overall_counter[color]
        continuity_bonus = continuity_scores.get(color, 0) * 2
        edge_bonus = edge_scores.get(color, 0) if col in (0, cols-1) or row in (0, 2) else 0
        section_relevance = section_counter[color] * 3
        structural_bonus = structural_scores.get(color, 0)
        return base_score + continuity_bonus + edge_bonus + section_relevance + structural_bonus

    output_values = []
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        section_counter = analyze_grid(section)
        
        output_row = []
        for col in range(cols):
            color_scores = {color: color_score(color, section_counter, i, col) for color in overall_counter}
            chosen_color = max(color_scores, key=color_scores.get)
            output_row.append(chosen_color)
        
        output_values.append(output_row)

    # Post-processing
    for col in range(cols):
        column = [row[col] for row in input_grid.values]
        if len(set(column)) == 1 and column[0] != 0:
            for i in range(3):
                output_values[i][col] = column[0]

    # Ensure representation of top colors from each section
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        section_counter = analyze_grid(section)
        top_colors = [color for color, _ in section_counter.most_common(3) if color != 0]
        for color in top_colors:
            if color not in output_values[i]:
                least_significant_index = min(range(cols), key=lambda c: color_score(output_values[i][c], section_counter, i, c))
                output_values[i][least_significant_index] = color

    # Preserve diagonal patterns
    for i in range(3):
        for j in range(cols):
            if i > 0 and j > 0 and input_grid.values[i*section_height-1][j-1] == input_grid.values[i*section_height][j]:
                output_values[i][j] = input_grid.values[i*section_height][j]

    return ColoredGrid(values=output_values)
