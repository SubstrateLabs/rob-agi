from rob_agi.colored_grid import ColoredGrid

def solve_5289ad53(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 2x3 output grid based on the presence and distribution of horizontal red and green lines.
    
    The solution works as follows:
    1. Detect all horizontal red and green lines in the input grid.
    2. Analyze the distribution of colors (red and green) and line lengths.
    3. Create a 2x3 output grid where:
       - Top row is always [3, 3, X], where X is 3 if there are more green lines than red lines, 2 otherwise.
       - Bottom row is [Y, Z, W], where:
         Y is 3 if there are any green lines, 2 if there are only red lines, 0 if no lines exist.
         Z is 2 if there are at least two different line lengths, 0 otherwise.
         W is 2 if there are at least three different line lengths, 0 otherwise.
    """
    # Step 1: Detect lines
    lines = input_grid.detect_lines()
    red_lines = [line for line in lines if line[0] == 2]
    green_lines = [line for line in lines if line[0] == 3]

    # Step 2: Analyze distribution of colors and line lengths
    num_red_lines = len(red_lines)
    num_green_lines = len(green_lines)
    unique_lengths = set(len(line[1]) for line in red_lines + green_lines)

    # Step 3: Create output grid
    top_right = 3 if num_green_lines > num_red_lines else 2
    bottom_left = 3 if num_green_lines > 0 else (2 if num_red_lines > 0 else 0)
    bottom_middle = 2 if len(unique_lengths) >= 2 else 0
    bottom_right = 2 if len(unique_lengths) >= 3 else 0

    output = [
        [3, 3, top_right],
        [bottom_left, bottom_middle, bottom_right]
    ]

    # Step 4: Return new ColoredGrid
    return ColoredGrid(values=output)
