from rob_agi.colored_grid import ColoredGrid
import copy

def solve_212895b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating colors from gray squares.
    
    For each gray (5) square, starting from bottom-right to top-left:
    1. Propagate red (2) color towards bottom-left.
    2. Propagate yellow (4) color towards top-right.
    3. Limit propagation to 4 steps from the origin.
    4. Continue propagation through gray squares.
    5. Stop propagation at grid edges or when encountering pre-existing colors.
    6. Yellow overwrites red, but not vice versa.
    7. Preserve original gray squares and other pre-existing colors.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid.values), len(grid.values[0])

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def propagate_red(r, c, step):
        if step > 4 or not is_valid(r, c) or grid.values[r][c] not in [0, 5]:
            return
        if grid.values[r][c] != 5:
            grid.values[r][c] = 2
        new_step = 0 if grid.values[r][c] == 5 else step + 1
        for dr, dc in [(-1, -1), (0, -1), (1, -1)]:
            propagate_red(r + dr, c + dc, new_step)

    def propagate_yellow(r, c, step):
        if step > 4 or not is_valid(r, c) or grid.values[r][c] not in [0, 2, 5]:
            return
        if grid.values[r][c] != 5:
            grid.values[r][c] = 4
        new_step = 0 if grid.values[r][c] == 5 else step + 1
        for dr, dc in [(-1, 1), (0, 1), (1, 1)]:
            propagate_yellow(r + dr, c + dc, new_step)

    gray_squares = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5]
    gray_squares.sort(key=lambda x: (-x[0] - x[1], -x[0]))

    for r, c in gray_squares:
        propagate_red(r, c, 0)
        propagate_yellow(r, c, 0)

    # Restore original gray squares
    for r, c in gray_squares:
        grid.values[r][c] = 5

    return grid
