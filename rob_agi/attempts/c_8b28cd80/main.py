from rob_agi.colored_grid import ColoredGrid

def solve_8b28cd80(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by drawing a 7-segment digit.
    The position of the non-zero color in the input determines which digit to draw.
    The non-zero color is used to draw the digit in the output grid.
    """
    # Position to digit mapping
    position_to_digit = {
        (0, 0): 0, (0, 1): 1, (0, 2): 3,
        (1, 0): 4, (1, 1): 5, (1, 2): 6,
        (2, 0): 7, (2, 1): 8, (2, 2): 9
    }

    # Segment definitions for each digit
    segments = {
        0: [0, 1, 2, 3, 4, 5],
        1: [1, 2],
        2: [0, 1, 3, 4, 6],
        3: [0, 1, 2, 3, 6],
        4: [1, 2, 5, 6],
        5: [0, 2, 3, 5, 6],
        6: [0, 2, 3, 4, 5, 6],
        7: [0, 1, 2],
        8: [0, 1, 2, 3, 4, 5, 6],
        9: [0, 1, 2, 3, 5, 6]
    }

    def draw_segment(grid, segment, color):
        if segment == 0:
            for i in range(1, 8):
                grid[0][i] = color
        elif segment == 1:
            for i in range(1, 4):
                grid[i][8] = color
        elif segment == 2:
            for i in range(5, 8):
                grid[i][8] = color
        elif segment == 3:
            for i in range(1, 8):
                grid[8][i] = color
        elif segment == 4:
            for i in range(5, 8):
                grid[i][0] = color
        elif segment == 5:
            for i in range(1, 4):
                grid[i][0] = color
        elif segment == 6:
            for i in range(1, 8):
                grid[4][i] = color

    def draw_digit(digit, color):
        grid = [[0 for _ in range(9)] for _ in range(9)]
        for segment in segments[digit]:
            draw_segment(grid, segment, color)
        return grid

    # Find non-zero color and its position
    color = 0
    position = (0, 0)
    for i in range(3):
        for j in range(3):
            if input_grid.values[i][j] != 0:
                color = input_grid.values[i][j]
                position = (i, j)
                break
        if color != 0:
            break

    # Determine digit and draw it
    digit = position_to_digit[position]
    output_grid = draw_digit(digit, color)

    return ColoredGrid(values=output_grid)
