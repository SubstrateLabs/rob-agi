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
    color_origins = [[None for _ in range(cols)] for _ in range(rows)]
    propagation_limit = 4

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def get_next_positions(r, c, direction):
        if direction == 'down-left':
            return [(r+1, c-1), (r+1, c), (r, c-1)]
        else:  # 'up-right'
            return [(r-1, c+1), (r-1, c), (r, c+1)]

    gray_squares = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5]
    gray_squares.sort(key=lambda x: (-x[0] - x[1], -x[0]))

    for index, (r, c) in enumerate(gray_squares):
        for color, direction in [(2, 'down-left'), (4, 'up-right')]:
            queue = deque([(r, c, 0)])  # (row, col, step)
            while queue:
                curr_r, curr_c, step = queue.popleft()
                if not is_valid(curr_r, curr_c) or step > propagation_limit:
                    continue

                current_color = grid.values[curr_r][curr_c]
                if current_color not in [0, 5]:
                    if index > color_origins[curr_r][curr_c]:
                        grid.values[curr_r][curr_c] = color
                        color_origins[curr_r][curr_c] = index
                    continue

                if current_color == 0:
                    grid.values[curr_r][curr_c] = color
                    color_origins[curr_r][curr_c] = index

                for next_r, next_c in get_next_positions(curr_r, curr_c, direction):
                    if direction == 'down-left':
                        queue.append((next_r, next_c, step + 1 if next_r > curr_r and next_c < curr_c else step))
                    else:  # 'up-right'
                        queue.append((next_r, next_c, step + 1 if next_r < curr_r and next_c > curr_c else step))

    # Restore original gray squares
    for r, c in gray_squares:
        grid.values[r][c] = 5

    return grid
