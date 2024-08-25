from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_9caba7c3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Preserves red (2) squares in the upper-left quadrant.
    2. Changes other red (2) squares to yellow (4).
    3. Propagates orange (7) color from yellow squares to the right and below,
       replacing gray (5) or red (2) squares.
    4. Preserves the original state of other colors.
    5. Applies transformations based on the original grid state.
    """
    original_grid = input_grid.deep_copy()
    new_grid = input_grid.deep_copy()
    height, width = len(original_grid.values), len(original_grid.values[0])
    mid_row, mid_col = height // 2, width // 2

    def is_upper_left_quadrant(row, col):
        return row < mid_row and col < mid_col

    def propagate_orange(start_row, start_col):
        queue = deque([(start_row, start_col)])
        while queue:
            row, col = queue.popleft()
            for dr, dc in [(0, 1), (1, 0)]:  # Right and below
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < height and 0 <= new_col < width and
                    original_grid.values[new_row][new_col] in [2, 5] and
                    new_grid.values[new_row][new_col] != 7):
                    new_grid.values[new_row][new_col] = 7
                    queue.append((new_row, new_col))

    yellow_squares = []
    for row in range(height):
        for col in range(width):
            if original_grid.values[row][col] == 2:
                if not is_upper_left_quadrant(row, col):
                    new_grid.values[row][col] = 4
                    yellow_squares.append((row, col))

    for row, col in yellow_squares:
        propagate_orange(row, col)

    return new_grid
