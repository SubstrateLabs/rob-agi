from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5289ad53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 2x3 output grid based on the presence and length of horizontal red and green lines.
    
    The solution works as follows:
    1. Detect all horizontal red and green lines in the input grid.
    2. Analyze the presence of colors (red and green) and line lengths.
    3. Create a 2x3 output grid where:
       - Top row is always [3, 3, X], where X is 3 if the longest green line is longer than the longest red line,
         2 if red lines are present and no green line is longer, 0 otherwise.
       - Bottom row is [Y, Z, W], where:
         Y is 2 if any red lines exist, 3 if only green lines exist, 0 if no lines exist.
         Z is 2 if any lines (red or green) of length exactly 4 exist, 0 otherwise.
         W is 2 if any lines (red or green) of length 5 or more exist, 0 otherwise.
    """
    # Step 1: Detect lines
    lines = input_grid.detect_lines()
    red_lines = [line for line in lines if line[0] == 2]
    green_lines = [line for line in lines if line[0] == 3]

    # Step 2: Analyze presence of colors and line lengths
    has_red = len(red_lines) > 0
    has_green = len(green_lines) > 0
    longest_red = max([len(line[1]) for line in red_lines], default=0)
    longest_green = max([len(line[1]) for line in green_lines], default=0)
    has_line_4 = any(len(line[1]) == 4 for line in red_lines + green_lines)
    has_line_5_or_more = any(len(line[1]) >= 5 for line in red_lines + green_lines)

    # Step 3: Create output grid
    top_right = 3 if longest_green > longest_red else (2 if has_red else 0)
    bottom_left = 2 if has_red else (3 if has_green else 0)
    bottom_middle = 2 if has_line_4 else 0
    bottom_right = 2 if has_line_5_or_more else 0

    output = [
        [3, 3, top_right],
        [bottom_left, bottom_middle, bottom_right]
    ]

    # Step 4: Return new ColoredGrid
    return ColoredGrid(values=output)
