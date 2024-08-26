from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def get_most_frequent_color(section: list[list[int]]) -> int:
    """
    Determine the most frequent color in a 4x4 section.
    In case of a tie, return the color with the lower numeric value.
    """
    flat_section = [color for row in section for color in row]
    color_counts = Counter(flat_section)
    return min(color_counts.items(), key=lambda x: (-x[1], x[0]))[0]

def solve_67b4a34d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the challenge by compressing a 16x16 input grid into a 4x4 output grid.
    
    The solution divides the input grid into sixteen 4x4 sections and determines
    the most frequent color in each section. This color is then used to fill
    the corresponding cell in the output grid. In case of ties, the color with
    the lower numeric value is chosen.
    
    If multiple colors have the same highest frequency in a section, the solution
    chooses the color that appears first in the section when read from left to right,
    top to bottom.
    
    Args:
    input_grid (ColoredGrid): A 16x16 input grid
    
    Returns:
    ColoredGrid: A 4x4 grid representing the compressed input
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 16 or cols != 16:
        raise ValueError("Input grid must be 16x16")
    
    output_values = []
    for i in range(0, 16, 4):
        output_row = []
        for j in range(0, 16, 4):
            section = [row[j:j+4] for row in input_grid.values[i:i+4]]
            flat_section = [color for row in section for color in row]
            color_counts = Counter(flat_section)
            max_count = max(color_counts.values())
            most_frequent_colors = [color for color, count in color_counts.items() if count == max_count]
            output_row.append(min(most_frequent_colors, key=flat_section.index))
        output_values.append(output_row)
    
    return ColoredGrid(values=output_values)
