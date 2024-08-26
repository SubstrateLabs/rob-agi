from rob_agi.colored_grid import ColoredGrid

def solve_e6de6e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 2x12 input grid into an 8x7 output grid based on the pattern of red squares.
    
    The function works as follows:
    1. Initialize an 8x7 grid with black (0) squares and a green (3) square at the top center.
    2. Find all positions of red squares in the top input row.
    3. Calculate branch lengths for each red square position in the top row.
    4. Draw the branch starting from the center, moving down and right:
       a. Move down for each branch length.
       b. Move right after processing each red square position.
       c. Continue straight down after the last red square position.
    5. Return the completed output grid.
    """
    output = [[0 for _ in range(7)] for _ in range(8)]
    output[0][3] = 3  # Green square at top center

    top_red_positions = [i for i, v in enumerate(input_grid.values[0]) if v == 2]

    def get_branch_length(start_col: int) -> int:
        return sum(1 for col in range(start_col, len(input_grid.values[1])) if input_grid.values[1][col] == 2)

    branch_lengths = [get_branch_length(pos) for pos in top_red_positions]

    current_row, current_col = 1, 3

    for i, length in enumerate(branch_lengths):
        # Move down
        for _ in range(length):
            if current_row < 8:
                output[current_row][current_col] = 2
                current_row += 1

        # Move right (if not the last position)
        if i < len(branch_lengths) - 1 and current_col < 6:
            current_col += 1
            output[current_row-1][current_col] = 2

    # Continue straight down to the bottom
    while current_row < 8:
        output[current_row][current_col] = 2
        current_row += 1

    return ColoredGrid(values=output)
