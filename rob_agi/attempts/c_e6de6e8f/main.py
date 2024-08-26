from rob_agi.colored_grid import ColoredGrid

def solve_e6de6e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 2x12 input grid into an 8x7 output grid based on the pattern of red squares.
    
    The function works as follows:
    1. Initialize an 8x7 grid with black (0) squares and a green (3) square at the top center.
    2. Map horizontal positions in the input to vertical start positions in the output.
    3. For each red square in the top input row:
       a. Calculate the starting row in the output using the mapping function.
       b. Count consecutive red squares in the bottom row at the corresponding position.
       c. Create a diagonal sequence of red squares in the output grid, moving down and right.
    4. Return the completed output grid.
    """
    output = [[0 for _ in range(7)] for _ in range(8)]
    output[0][3] = 3  # Place green square at top center

    def map_position(i: int) -> int:
        return min(7, max(0, round(i * 7 / 11)))

    current_column = 0
    for i, top_value in enumerate(input_grid.values[0]):
        if top_value == 2:  # Red square in top row
            starting_row = map_position(i)
            length = 0
            for j in range(i, len(input_grid.values[1])):
                if input_grid.values[1][j] == 2:
                    length += 1
                else:
                    break
            
            for j in range(length):
                if current_column < 7 and starting_row + j < 8:
                    output[starting_row + j][current_column] = 2
                else:
                    break
            current_column += 1

    return ColoredGrid(values=output)
