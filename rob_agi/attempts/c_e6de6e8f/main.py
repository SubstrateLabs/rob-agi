from rob_agi.colored_grid import ColoredGrid

def solve_e6de6e8f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 2x12 input grid into an 8x7 output grid based on the pattern of red squares.
    
    The function works as follows:
    1. Initialize an 8x7 grid with black (0) squares and a green (3) square at the top center.
    2. Find all positions of red squares in the top input row as decision points.
    3. Process each decision point:
       a. If the corresponding position in the bottom row is red, go straight down.
       b. If not, go diagonally down-right.
    4. Continue the path to the bottom of the grid.
    5. Return the completed output grid.
    """
    output = [[0 for _ in range(7)] for _ in range(8)]
    output[0][3] = 3  # Green square at top center

    decision_points = [i for i, v in enumerate(input_grid.values[0]) if v == 2]
    current_row, current_col = 1, 3

    for i, decision_point in enumerate(decision_points):
        go_straight = input_grid.values[1][decision_point] == 2

        while (i < len(decision_points) - 1 and current_col < decision_points[i+1]) or \
              (i == len(decision_points) - 1 and current_row < 8):
            output[current_row][current_col] = 2
            current_row += 1
            if not go_straight and current_col < 6:
                current_col += 1

        if i < len(decision_points) - 1 and current_col < 6:
            current_col += 1

    # Ensure the path reaches the bottom
    while current_row < 8:
        output[current_row][current_col] = 2
        current_row += 1

    return ColoredGrid(values=output)
