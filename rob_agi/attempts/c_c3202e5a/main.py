from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_c3202e5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid by identifying dividing lines, focus color, and creating a new grid
    based on the pattern of the focus color.

    1. Identifies the dividing lines in the input grid.
    2. Determines the section size (3x3 or 4x4).
    3. Finds the focus color (most frequent non-dividing, non-black color).
    4. Analyzes the distribution of the focus color in each section's quadrants.
    5. Creates a new grid (5x5 if input sections are 3x3, 3x3 if input sections are 4x4).
    6. Applies a transformation rule to place the focus color in the output grid based on the most frequent quadrant patterns.

    The transformation captures the essence of the focus color's distribution
    in a simplified geometric pattern, such as an L-shape, diagonal line, or corner pattern.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed output grid.
    """
    dividing_color = find_dividing_lines(input_grid)
    section_size = get_section_size(input_grid, dividing_color)
    focus_color = get_focus_color(input_grid, dividing_color)
    
    quadrant_patterns = analyze_quadrant_patterns(input_grid, dividing_color, focus_color, section_size)
    
    if section_size == 3:
        output_size = 5
        output_values = expand_pattern(quadrant_patterns, focus_color)
    elif section_size == 4:
        output_size = 3
        output_values = contract_pattern(quadrant_patterns, focus_color)
    else:
        raise ValueError(f"Unexpected section size: {section_size}")

    return ColoredGrid(values=output_values)

def find_dividing_lines(grid: ColoredGrid) -> int:
    """Identifies the color of the dividing lines."""
    rows, cols = grid.get_dimensions()
    for row in range(rows):
        if all(cell == grid.values[row][0] for cell in grid.values[row]) and grid.values[row][0] != 0:
            return grid.values[row][0]
    raise ValueError("No dividing lines found")

def get_section_size(grid: ColoredGrid, dividing_color: int) -> int:
    """Calculates the size of individual sections."""
    section_size = 0
    for row in grid.values:
        if row[0] == dividing_color:
            return section_size
        section_size += 1
    raise ValueError("Could not determine section size")

def get_focus_color(grid: ColoredGrid, dividing_color: int) -> int:
    """Determines the most frequent non-zero, non-dividing color in the grid."""
    color_counts = {color: 0 for color in range(1, 10) if color != dividing_color}
    for row in grid.values:
        for cell in row:
            if cell != 0 and cell != dividing_color:
                color_counts[cell] = color_counts.get(cell, 0) + 1
    return max(color_counts, key=color_counts.get)

def analyze_quadrant_patterns(grid: ColoredGrid, dividing_color: int, focus_color: int, section_size: int) -> Dict[Tuple[int, int], int]:
    """Analyzes the distribution of the focus color in each section's quadrants."""
    quadrant_patterns = {}
    sections = [row for row in grid.values if row[0] != dividing_color]
    
    for i in range(0, len(sections), section_size):
        for j in range(0, len(sections[0]), section_size):
            section = [row[j:j+section_size] for row in sections[i:i+section_size]]
            quadrants = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
            
            for r, row in enumerate(section):
                for c, cell in enumerate(row):
                    if cell == focus_color:
                        quadrants[(r//(section_size//2), c//(section_size//2))] += 1
            
            sorted_quadrants = tuple(sorted(quadrants.items(), key=lambda x: x[1], reverse=True)[:2])
            quadrant_patterns[sorted_quadrants] = quadrant_patterns.get(sorted_quadrants, 0) + 1
    
    return quadrant_patterns

def expand_pattern(quadrant_patterns: Dict[Tuple[int, int], int], focus_color: int) -> List[List[int]]:
    """Expands the pattern from 3x3 sections to a 5x5 grid."""
    output = [[0 for _ in range(5)] for _ in range(5)]
    most_common_pattern = max(quadrant_patterns, key=quadrant_patterns.get)
    
    if most_common_pattern[0][0] == most_common_pattern[1][0]:
        # Adjacent quadrants - L-shape
        q = most_common_pattern[0][0]
        output[q[0]*4][q[1]*4] = focus_color
        output[q[0]*4][2] = focus_color
        output[2][q[1]*4] = focus_color
        output[4-q[0]*4][4-q[1]*4] = focus_color
        output[2][2] = focus_color
    else:
        # Diagonal quadrants
        output[0][0] = focus_color
        output[0][4] = focus_color
        output[4][0] = focus_color
        output[4][4] = focus_color
        output[2][2] = focus_color
    
    return output

def contract_pattern(quadrant_patterns: Dict[Tuple[int, int], int], focus_color: int) -> List[List[int]]:
    """Contracts the pattern from 4x4 sections to a 3x3 grid."""
    output = [[0 for _ in range(3)] for _ in range(3)]
    most_common_pattern = max(quadrant_patterns, key=quadrant_patterns.get)
    
    # Determine the dominant diagonal
    if (most_common_pattern[0][0] == (0,0) and most_common_pattern[1][0] == (1,1)) or \
       (most_common_pattern[0][0] == (1,1) and most_common_pattern[1][0] == (0,0)):
        output[0][0] = focus_color
        output[2][2] = focus_color
    else:
        output[0][2] = focus_color
        output[2][0] = focus_color
    
    output[1][1] = focus_color  # Always fill the center
    
    return output
