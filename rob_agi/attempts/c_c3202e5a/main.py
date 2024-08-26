from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List, Dict

def solve_c3202e5a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a simplified output grid based on color patterns.

    1. Identifies the dividing color that forms continuous lines in the input grid.
    2. Determines the section layout and calculates section size.
    3. Identifies the focus color by analyzing color distribution within sections.
    4. Analyzes the focus color's placement patterns across all sections.
    5. Generates an output grid (5x5 for 4x4 or 5x5 input sections).
    6. Translates the focus color pattern into a representative shape in the output grid.

    The transformation simplifies complex input patterns into a representative geometric shape,
    capturing the essence of the focus color's distribution in the input grid.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The simplified output grid representing the essence of the input pattern.
    """
    dividing_color = find_dividing_lines(input_grid)
    section_size = get_section_size(input_grid, dividing_color)
    focus_color = get_focus_color(input_grid, dividing_color)
    
    section_patterns = analyze_section_patterns(input_grid, dividing_color, focus_color, section_size)
    
    output_values = generate_output_pattern(section_patterns, focus_color)

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

def analyze_section_patterns(grid: ColoredGrid, dividing_color: int, focus_color: int, section_size: int) -> List[List[int]]:
    """Analyzes the distribution of the focus color in each section."""
    sections = [row for row in grid.values if row[0] != dividing_color]
    pattern = []
    
    for i in range(0, len(sections), section_size):
        row_pattern = []
        for j in range(0, len(sections[0]), section_size):
            section = [row[j:j+section_size] for row in sections[i:i+section_size]]
            focus_count = sum(cell == focus_color for row in section for cell in row)
            row_pattern.append(focus_count)
        pattern.append(row_pattern)
    
    return pattern

def find_dividing_lines(grid: ColoredGrid) -> int:
    """Identifies the color of the dividing lines."""
    rows, cols = grid.get_dimensions()
    for color in range(1, 10):
        if any(all(cell == color for cell in row) for row in grid.values):
            return color
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
    color_counts = {}
    for row in grid.values:
        for cell in row:
            if cell != 0 and cell != dividing_color:
                color_counts[cell] = color_counts.get(cell, 0) + 1
    return max(color_counts, key=color_counts.get)
def generate_output_pattern(section_patterns: List[List[int]], focus_color: int) -> List[List[int]]:
    """Generates the output pattern based on the section patterns."""
    output = [[0 for _ in range(5)] for _ in range(5)]
    max_count = max(max(row) for row in section_patterns)
    threshold = max_count // 2

    for i, row in enumerate(section_patterns):
        for j, count in enumerate(row):
            if count > threshold:
                output[i][j] = focus_color

    # Ensure at least one cell is filled
    if sum(sum(row) for row in output) == 0:
        max_pos = max(((i, j) for i, row in enumerate(section_patterns) for j, count in enumerate(row)),
                      key=lambda pos: section_patterns[pos[0]][pos[1]])
        output[max_pos[0]][max_pos[1]] = focus_color

    return output
