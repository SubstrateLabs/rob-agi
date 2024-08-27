from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Dict

def solve_3d31c5b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 12x6 input grid into a 3x6 output grid by analyzing and condensing color patterns.
    
    The function divides the input into three 4x6 sections and analyzes them along with
    overlapping 5x6 sections. It preserves key patterns, maintains vertical continuity,
    and balances color frequencies while adhering to specific rules for edges.
    
    Key steps:
    1. Analyze the input grid using both fixed and overlapping sections
    2. Preserve edges based on specific rules
    3. Detect and apply vertical and alternating patterns
    4. Balance color frequencies with positional weighting
    5. Ensure vertical continuity and pattern consistency
    6. Generate the output grid considering all analyzed factors
    7. Post-process to ensure representation of dominant colors
    
    Args:
    input_grid (ColoredGrid): A 12x6 grid representing the input pattern
    
    Returns:
    ColoredGrid: A 3x6 grid representing the transformed output pattern
    """
    rows, cols = input_grid.get_dimensions()
    section_height = rows // 3

    def analyze_section(section: List[List[int]], weights: List[float]) -> Dict[int, float]:
        counter = Counter()
        for row, weight in zip(section, weights):
            counter.update({cell: weight for cell in row if cell != 0})
        return dict(counter)

    def get_vertical_pattern(grid: List[List[int]], col: int) -> List[int]:
        return [row[col] for row in grid if row[col] != 0]

    def detect_alternating_pattern(sequence: List[int]) -> List[int]:
        if len(sequence) < 2:
            return sequence
        if len(set(sequence[::2])) == 1 and len(set(sequence[1::2])) == 1:
            return sequence[:2]
        return []

    # Analyze fixed sections and overlapping windows
    section_analyses = []
    overlapping_analyses = []
    for i in range(3):
        section = input_grid.values[i*section_height:(i+1)*section_height]
        section_analyses.append(analyze_section(section, [1.0, 1.0, 1.0, 1.0]))
        
        if i < 2:
            window = input_grid.values[i*section_height:(i+2)*section_height]
            overlapping_analyses.append(analyze_section(window, [0.5, 1.0, 1.0, 1.0, 1.0]))

    output_values = [[0 for _ in range(cols)] for _ in range(3)]

    # Preserve edges
    for i in range(3):
        output_values[i][0] = input_grid.values[i*section_height][0]
        output_values[i][-1] = input_grid.values[i*section_height + section_height - 1][-1]

    # Generate output grid
    for i in range(3):
        for j in range(1, cols-1):
            color_scores = {}
            
            # Consider section analysis
            for color, score in section_analyses[i].items():
                color_scores[color] = color_scores.get(color, 0) + score * 2
            
            # Consider overlapping analysis
            if i < 2:
                for color, score in overlapping_analyses[i].items():
                    color_scores[color] = color_scores.get(color, 0) + score
            
            # Vertical continuity
            vertical_pattern = get_vertical_pattern(input_grid.values, j)
            for color in vertical_pattern:
                color_scores[color] = color_scores.get(color, 0) + vertical_pattern.count(color) * 0.5
            
            # Detect alternating patterns
            alt_pattern = detect_alternating_pattern(vertical_pattern)
            if alt_pattern:
                color_scores[alt_pattern[i % len(alt_pattern)]] += 2
            
            # Choose color
            chosen_color = max(color_scores, key=color_scores.get) if color_scores else 0
            output_values[i][j] = chosen_color

    # Post-processing: Ensure dominant colors are represented
    for i in range(3):
        dominant_colors = sorted(section_analyses[i].items(), key=lambda x: x[1], reverse=True)[:2]
        for dominant_color, _ in dominant_colors:
            if dominant_color not in output_values[i]:
                min_score_index = min(range(1, cols-1), key=lambda x: section_analyses[i].get(output_values[i][x], 0))
                output_values[i][min_score_index] = dominant_color

    return ColoredGrid(values=output_values)
