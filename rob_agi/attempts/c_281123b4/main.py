from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a two-step process:
    1. Row processing: Divides each row into 4 sections, choosing the most frequent non-zero color
       (or the highest value color in case of a tie) for each section.
    2. Column processing: For each column in the resulting 4x4 grid, selects the most frequent
       non-zero color (or the highest value color in case of a tie) and replaces any zero values
       in that column with the selected color.
    The final 4x4 grid is then returned.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    def process_section(section: List[int]) -> int:
        color_counts = Counter(color for color in section if color != 0)
        if not color_counts:
            return 0
        return max(color_counts.items(), key=lambda x: (x[1], x[0]))[0]

    # Row processing
    intermediate_values = []
    for row in input_grid.values:
        sections = [row[0:5], row[5:10], row[10:15], row[15:19]]
        new_row = [process_section(section) for section in sections]
        intermediate_values.append(new_row)

    # Column processing
    for col in range(4):
        column = [row[col] for row in intermediate_values]
        most_frequent_color = process_section(column)
        for row in range(4):
            if intermediate_values[row][col] == 0:
                intermediate_values[row][col] = most_frequent_color

    return ColoredGrid(values=intermediate_values)
