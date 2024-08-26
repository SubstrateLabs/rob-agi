from rob_agi.colored_grid import ColoredGrid

def solve_e6de6e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 2x12 input grid into an 8x7 output grid based on the pattern of red squares.
    
    The function works as follows:
    1. Initialize an 8x7 grid with black (0) squares and a green (3) square at the top center.
    2. Find all positions of red squares in the top input row and the rightmost red square position.
    3. Map input columns to output columns using a mapping function.
    4. For each red square in the top input row:
       a. Calculate the branch length by counting consecutive red squares in the bottom row.
       b. Draw a branch in the output grid, starting from the mapped column and moving down and right.
    5. Return the completed output grid.
    """
    def map_column(input_col: int) -> int:
        return min(6, round(input_col * 6 / 11))

    def get_branch_length(start_col: int) -> int:
        length = 0
        for col in range(start_col, len(input_grid.values[1])):
            if input_grid.values[1][col] == 2:
                length += 1
            else:
                break
        return length

    def draw_branch(start_col: int, length: int, max_right_col: int) -> None:
        row, col = 1, start_col
        for _ in range(length):
            if row >= 8 or col > max_right_col:
                break
            output[row][col] = 2
            row += 1
            if col < max_right_col:
                col += 1

    output = [[0 for _ in range(7)] for _ in range(8)]
    output[0][3] = 3  # Green square at top center

    top_red_positions = [i for i, v in enumerate(input_grid.values[0]) if v == 2]
    max_right_col = map_column(max(top_red_positions))

    for pos in top_red_positions:
        start_col = map_column(pos)
        length = get_branch_length(pos)
        draw_branch(start_col, length, max_right_col)

    return ColoredGrid(values=output)
