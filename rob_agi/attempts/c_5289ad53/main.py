from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5289ad53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 2x3 output grid based on the presence and length of horizontal red and green lines.
    
    The solution works as follows:
    1. Detect all horizontal red and green lines in the input grid.
    2. Categorize lines by length: short (1-3), medium (4-6), and long (7+).
    3. Analyze the presence of colors (red and green) and line lengths.
    4. Create a 2x3 output grid where:
       - Top row is always [3, 3, X], where X is 2 if red lines are present, 0 otherwise.
       - Bottom row is [Y, Z, W], where:
         Y is 2 if any red lines exist, 3 if only green lines exist, 0 if no lines exist.
         Z is 2 if medium lines exist (of either color), 0 otherwise.
         W is 2 if long lines exist (of either color), 0 otherwise.
    """
    # Step 1: Detect lines
    lines = input_grid.detect_lines()
    color_lines = [line for line in lines if line[0] in [2, 3]]  # Keep only red and green lines

    # Step 2: Categorize lines
    medium_lines, long_lines = [], []
    for color, coords in color_lines:
        length = len(coords)
        if 4 <= length <= 6:
            medium_lines.append((color, coords))
        elif length >= 7:
            long_lines.append((color, coords))

    # Step 3: Analyze presence of colors and line lengths
    has_red = any(line[0] == 2 for line in color_lines)
    has_green = any(line[0] == 3 for line in color_lines)
    has_medium = len(medium_lines) > 0
    has_long = len(long_lines) > 0

    # Step 4: Create output grid
    output = [
        [3, 3, 2 if has_red else 0],
        [2 if has_red else (3 if has_green else 0), 2 if has_medium else 0, 2 if has_long else 0]
    ]

    # Step 5: Return new ColoredGrid
    return ColoredGrid(values=output)
