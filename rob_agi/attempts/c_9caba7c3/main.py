from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_9caba7c3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Preserves red (2) squares in the upper-left quadrant.
    2. Identifies connected red regions outside the upper-left quadrant.
    3. Changes these identified red regions to yellow (4).
    4. Propagates orange (7) color within the boundaries of the originally red regions,
       moving right and down, but not crossing black (0) squares.
    5. Preserves the original state of black (0) squares and all other colors outside the red regions.
    6. Applies transformations based on the original grid state.
    """
    original_grid = input_grid.deep_copy()
    new_grid = input_grid.deep_copy()
    height, width = len(original_grid.values), len(original_grid.values[0])
    mid_row, mid_col = height // 2, width // 2

    def is_upper_left_quadrant(row, col):
        return row < mid_row and col < mid_col

    def flood_fill(row, col, color, target_color, visited):
        if (row < 0 or row >= height or col < 0 or col >= width or
            original_grid.values[row][col] != target_color or (row, col) in visited):
            return

        visited.add((row, col))
        new_grid.values[row][col] = color

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            flood_fill(row + dr, col + dc, color, target_color, visited)

    def propagate_orange(start_row, start_col, visited):
        queue = deque([(start_row, start_col)])
        while queue:
            row, col = queue.popleft()
            if (row, col) not in visited:
                continue
            new_grid.values[row][col] = 7
            for dr, dc in [(0, 1), (1, 0)]:  # Only right and down
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < height and 0 <= new_col < width and
                    original_grid.values[new_row][new_col] != 0 and
                    (new_row, new_col) in visited):
                    queue.append((new_row, new_col))

    visited = set()
    for row in range(height):
        for col in range(width):
            if original_grid.values[row][col] == 2 and not is_upper_left_quadrant(row, col):
                flood_fill(row, col, 4, 2, visited)

    for row in range(height):
        for col in range(width):
            if new_grid.values[row][col] == 4:
                propagate_orange(row, col, visited)

    # Final check for upper-left quadrant red squares
    for row in range(mid_row):
        for col in range(mid_col):
            if original_grid.values[row][col] == 2:
                new_grid.values[row][col] = 2

    return new_grid
