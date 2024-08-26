from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Section division: Divides each row into 4 sections, ignoring the green (3) separator column.
    2. Color analysis: Analyzes color frequencies in each section, prioritizing brown (9) and yellow (4).
    3. Pattern recognition: Identifies color patterns within and across sections.
    4. Output construction: Builds a 4x4 grid based on color priorities and patterns.
    5. Refinement: Adjusts the output to create balanced and symmetric patterns where possible.
    The final 4x4 grid is then returned, reflecting the most prominent colors and patterns from the input.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    def analyze_section(section: List[int]) -> Tuple[int, int]:
        color_counts = Counter(color for color in section if color != 0 and color != 3)
        if not color_counts:
            return 0, 0
        primary = max(color_counts.items(), key=lambda x: (x[1], x[0]))[0]
        secondary = max((c for c in color_counts if c != primary), default=0, key=lambda x: (color_counts[x], x))
        return primary, secondary

    # Section analysis
    sections = []
    for row in input_grid.values:
        row_sections = [row[0:5], row[5:10], row[10:15], row[15:19]]
        sections.append([analyze_section(section) for section in row_sections])

    # Pattern recognition and output construction
    output = [[0 for _ in range(4)] for _ in range(4)]
    for col in range(4):
        column_colors = [sections[row][col] for row in range(4)]
        primary_colors = [pair[0] for pair in column_colors if pair[0] != 0]
        secondary_colors = [pair[1] for pair in column_colors if pair[1] != 0]
        
        if primary_colors:
            main_color = max(set(primary_colors), key=primary_colors.count)
            alt_color = max(set(secondary_colors), key=secondary_colors.count) if secondary_colors else 0
            
            for row in range(4):
                if row % 2 == 0:
                    output[row][col] = main_color
                else:
                    output[row][col] = alt_color if alt_color != 0 else main_color

    # Refinement
    for row in range(4):
        row_colors = [c for c in output[row] if c != 0]
        if len(set(row_colors)) == 1 and 0 in output[row]:
            fill_color = max(set(output[row]) - {0})
            output[row] = [fill_color if c == 0 else c for c in output[row]]

    return ColoredGrid(values=output)
