from rob_agi.colored_grid import ColoredGrid

def solve_e6de6e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 2x12 input grid into an 8x7 output grid based on the pattern of red squares.
    
    The function works as follows:
    1. Initialize an 8x7 grid with black (0) squares and a green (3) square at the top center.
    2. For each red square in the top input row:
       a. Count consecutive red squares in the bottom row at the corresponding position.
       b. Create a "column" of red squares in the output grid, starting from the center.
       c. Shift right and extend downwards based on the count from step a.
    3. Return the completed output grid.
    """
    output = [[0 for _ in range(7)] for _ in range(8)]
    output[0][3] = 3  # Place green square at top center

    current_column = 3
    for i, top_value in enumerate(input_grid.values[0]):
        if top_value == 2:  # Red square in top row
            current_row = 1
            length = 0
            for j in range(i, len(input_grid.values[1])):
                if input_grid.values[1][j] == 2:
                    length += 1
                else:
                    break
            
            for _ in range(length):
                if current_row < 8 and current_column < 7:
                    output[current_row][current_column] = 2
                    current_row += 1
                    if _ > 0:  # Not the first square in the sequence
                        current_column += 1

    return ColoredGrid(values=output)
