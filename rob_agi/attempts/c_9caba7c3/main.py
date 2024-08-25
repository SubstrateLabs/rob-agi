from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_9caba7c3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Preserves red (2) squares in the upper-left quadrant.
    2. Changes other red (2) squares to yellow (4).
    3. Propagates orange (7) color from yellow squares to the right and below,
       replacing any non-black squares, potentially jumping over other colors.
    4. Preserves the original state of black (0) squares.
    5. Applies transformations based on the original grid state.
    """
    original_grid = input_grid.deep_copy()
    new_grid = input_grid.deep_copy()
    height, width = len(original_grid.values), len(original_grid.values[0])
    mid_row, mid_col = height // 2, width // 2

    def is_upper_left_quadrant(row, col):
        return row < mid_row and col < mid_col

    def propagate_orange(start_row, start_col):
        right_queue = deque([(start_row, start_col)])
        down_queue = deque([(start_row, start_col)])

        while right_queue or down_queue:
            if right_queue:
                row, col = right_queue.popleft()
                new_col = col
                while new_col < width and original_grid.values[row][new_col] != 0:
                    if not is_upper_left_quadrant(row, new_col) or original_grid.values[row][new_col] != 2:
                        new_grid.values[row][new_col] = 7
                    if row >= mid_row:
                        down_queue.append((row, new_col))
                    new_col += 1

            if down_queue:
                row, col = down_queue.popleft()
                new_row = row
                while new_row < height and original_grid.values[new_row][col] != 0:
                    if not is_upper_left_quadrant(new_row, col) or original_grid.values[new_row][col] != 2:
                        new_grid.values[new_row][col] = 7
                    if col >= mid_col:
                        right_queue.append((new_row, col))
                    new_row += 1

    yellow_squares = []
    for row in range(height):
        for col in range(width):
            if original_grid.values[row][col] == 2:
                if not is_upper_left_quadrant(row, col):
                    new_grid.values[row][col] = 4
                    yellow_squares.append((row, col))

    for row, col in yellow_squares:
        propagate_orange(row, col)

    # Final check for upper-left quadrant red squares
    for row in range(mid_row):
        for col in range(mid_col):
            if original_grid.values[row][col] == 2:
                new_grid.values[row][col] = 2

    return new_grid
