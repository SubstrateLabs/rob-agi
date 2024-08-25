from rob_agi.colored_grid import ColoredGrid
import copy
from collections import deque

def solve_212895b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating colors from gray squares.
    
    For each gray (5) square, starting from bottom-right to top-left:
    1. Propagate red (2) color towards bottom-left.
    2. Propagate yellow (4) color towards top-right.
    3. Limit propagation to 4 steps from the origin.
    4. Continue propagation through gray squares.
    5. Stop propagation at grid edges or when encountering pre-existing colors.
    6. Resolve color conflicts based on the order of original gray squares.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid.values), len(grid.values[0])

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def get_propagation_directions(color):
        return [(-1, 1), (0, 1), (1, 1)] if color == 2 else [(1, -1), (1, 0), (1, 1)]

    def propagate(start_r, start_c, color):
        queue = deque([(start_r, start_c, 0)])  # (row, col, steps)
        while queue:
            r, c, steps = queue.popleft()
            if not is_valid(r, c) or steps > 4:
                continue
            current_color = grid.values[r][c]
            if current_color not in [0, 5]:
                continue
            grid.values[r][c] = color
            new_steps = 0 if current_color == 5 else steps + 1
            for dr, dc in get_propagation_directions(color):
                queue.append((r + dr, c + dc, new_steps))

    gray_squares = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5]
    gray_squares.sort(key=lambda x: (-x[0] - x[1], -x[0]))

    for r, c in gray_squares:
        propagate(r, c, 2)  # Red
        propagate(r, c, 4)  # Yellow

    # Restore original gray squares
    for r, c in gray_squares:
        grid.values[r][c] = 5

    return grid
