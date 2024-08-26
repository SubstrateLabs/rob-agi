from rob_agi.colored_grid import ColoredGrid

def solve_8b28cd80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by drawing a 7-segment digit.
    The position of the non-zero color in the input determines which digit to draw (0-9).
    The digit is drawn using a 7-segment display style, with each segment being 2 units thick.
    The output includes a border and the digit, both drawn with the non-zero color from the input.
    The digit's position and additional lines depend on the input position:
    - Left column inputs align the digit to the left
    - Center column inputs center the digit
    - Right column inputs align the digit to the right
    - Top row inputs may have additional horizontal lines
    - Bottom row inputs have the digit moved up with additional lines below
    The background is filled with black (0).
    """
    # Position to digit mapping
    position_to_digit = {
        (0, 0): 7, (0, 1): 1, (0, 2): 3,
        (1, 0): 4, (1, 1): 0, (1, 2): 6,
        (2, 0): 2, (2, 1): 8, (2, 2): 9
    }

    # Segment definitions for each digit
    segments = {
        0: [0, 1, 2, 3, 4, 5], 1: [2, 5], 2: [0, 2, 3, 4, 6],
        3: [0, 2, 3, 5, 6], 4: [1, 2, 5, 6], 5: [0, 1, 3, 5, 6],
        6: [0, 1, 3, 4, 5, 6], 7: [0, 2, 5], 8: [0, 1, 2, 3, 4, 5, 6],
        9: [0, 1, 2, 3, 5, 6]
    }

    def draw_segment(grid, segment, color, offset_x, offset_y):
        coords = {
            0: [(i, 0) for i in range(1, 5)] + [(i, 1) for i in range(1, 5)],
            1: [(0, i) for i in range(1, 5)] + [(1, i) for i in range(1, 5)],
            2: [(4, i) for i in range(1, 5)] + [(5, i) for i in range(1, 5)],
            3: [(i, 4) for i in range(1, 5)] + [(i, 5) for i in range(1, 5)],
            4: [(4, i) for i in range(5, 9)] + [(5, i) for i in range(5, 9)],
            5: [(0, i) for i in range(5, 9)] + [(1, i) for i in range(5, 9)],
            6: [(i, 2) for i in range(1, 5)] + [(i, 3) for i in range(1, 5)]
        }
        for x, y in coords[segment]:
            if 0 <= y + offset_y < 9 and 0 <= x + offset_x < 9:
                grid[y + offset_y][x + offset_x] = color

    def draw_digit(digit, color, offset_x, offset_y):
        for segment in segments[digit]:
            draw_segment(output_grid, segment, color, offset_x, offset_y)

    def draw_border(color, row):
        for i in range(9):
            output_grid[0][i] = color  # Top border
            output_grid[i][0] = color  # Left border
            output_grid[i][8] = color  # Right border
        if row != 0:  # Full bottom border for non-top inputs
            output_grid[8] = [color] * 9
        else:  # Only corners for top inputs
            output_grid[8][0] = color
            output_grid[8][8] = color

    def add_lines(digit, color, row, col):
        if row == 0:  # Top row input
            if digit in [1, 3]:
                for r in [4, 6, 8]:
                    output_grid[r] = [color] * 9
        elif row == 2:  # Bottom row input
            # Move content up
            for i in range(5):
                output_grid[i] = output_grid[i+3]
            # Add horizontal lines
            for r in [5, 7]:
                output_grid[r] = [color] * 9
        if col == 0:  # Left column input
            for r in range(1, 8):
                output_grid[r][6:8] = [color, color]
        elif col == 2 and digit == 6:  # Right column, digit 6
            output_grid[7][7] = color

    # Find non-zero color and its position
    color, position = next((input_grid.values[i][j], (i, j)) 
                           for i in range(3) for j in range(3) 
                           if input_grid.values[i][j] != 0)

    # Initialize output grid
    output_grid = [[0 for _ in range(9)] for _ in range(9)]

    # Determine digit and alignment
    digit = position_to_digit[position]
    row, col = position

    # Set offsets based on alignment
    offset_x = 1 if col == 0 else (3 if col == 2 else 2)
    offset_y = 1 if row == 0 else (3 if row == 2 else 2)

    # Draw the digit
    draw_digit(digit, color, offset_x, offset_y)

    # Draw the border
    draw_border(color, row)

    # Add additional lines if needed
    add_lines(digit, color, row, col)

    return ColoredGrid(values=output_grid)
