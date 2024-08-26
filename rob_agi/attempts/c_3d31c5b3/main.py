from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple, Dict

def solve_3d31c5b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 12x6 input grid into a 3x6 output grid by analyzing and condensing color patterns.
    
    The function divides the input into three 4x6 sections and creates a condensed representation in the output.
    Each row of the output represents its corresponding section of the input, while also
    considering influences from other sections for continuity and pattern preservation.
    
    Key steps:
    1. Analyze each third of the input grid for color frequency and patterns
    2. Generate each output row based on its corresponding input section
    3. Preserve dominant colors and patterns from each section
    4. Maintain vertical continuity and color balance
    5. Apply special rules for edges and unique patterns
    
    Args:
    input_grid (ColoredGrid): A 12x6 grid representing the input pattern
    
    Returns:
    ColoredGrid: A 3x6 grid representing the transformed output pattern
    """
    rows, cols = input_grid.get_dimensions()
    section_height = rows // 3

    def analyze_section(section: List[List[int]]) -> Counter:
        return Counter(cell for row in section for cell in row if cell != 0)

    def get_vertical_pattern(grid: List[List[int]], col: int) -> List[int]:
        return [row[col] for row in grid if row[col] != 0]

    output_values = []
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        section_counter = analyze_section(section)
        
        output_row = []
        for col in range(cols):
            vertical_pattern = get_vertical_pattern(input_grid.values, col)
            
            color_scores = {}
            for color in set(section_counter.keys()) | set(vertical_pattern):
                score = section_counter[color] * 3  # Primary weight to section colors
                score += vertical_pattern.count(color) * 2  # Secondary weight to vertical continuity
                color_scores[color] = score
            
            chosen_color = max(color_scores, key=color_scores.get) if color_scores else 0
            output_row.append(chosen_color)
        
        output_values.append(output_row)

    # Preserve dominant colors in each section
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        section_counter = analyze_section(section)
        dominant_color = section_counter.most_common(1)[0][0] if section_counter else 0
        if dominant_color not in output_values[i]:
            least_common = min(output_values[i], key=output_values[i].count)
            output_values[i][output_values[i].index(least_common)] = dominant_color

    # Maintain vertical continuity
    for col in range(cols):
        column = [row[col] for row in input_grid.values]
        if len(set(column)) == 1 and column[0] != 0:
            for i in range(3):
                output_values[i][col] = column[0]

    # Special rules for edges
    for i in range(3):
        output_values[i][0] = input_grid.values[i*section_height][0]  # Left edge
        output_values[i][-1] = input_grid.values[i*section_height + section_height - 1][-1]  # Right edge

    # Preserve unique patterns
    for col in range(cols):
        unique_colors = set(input_grid.values[r][col] for r in range(rows) if input_grid.values[r][col] != 0)
        if len(unique_colors) == 1:
            unique_color = unique_colors.pop()
            for i in range(3):
                output_values[i][col] = unique_color

    return ColoredGrid(values=output_values)
