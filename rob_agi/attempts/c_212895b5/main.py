from rob_agi.colored_grid import ColoredGrid
import copy

def solve_212895b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by drawing diagonal lines from gray squares.
    
    For each gray (5) square, starting from bottom-right to top-left:
    1. Draw a red (2) diagonal line towards bottom-left until hitting an obstacle.
    2. Draw a yellow (4) diagonal line towards top-right until hitting an obstacle.
    3. Continue these lines even after encountering other gray squares.
    Obstacles include non-black squares (except gray), grid edges, or other colored lines.
    Lines can pass through gray squares without changing them.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid.values), len(grid.values[0])
    
    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols
    
    def draw_line(start_r, start_c, dr, dc, color):
        r, c = start_r + dr, start_c + dc  # Start from the next cell
        while is_valid(r, c):
            if grid.values[r][c] == 0:
                grid.values[r][c] = color
            elif grid.values[r][c] != 5:
                break
            r, c = r + dr, c + dc
    
    gray_squares = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5]
    gray_squares.sort(key=lambda x: (-x[0] - x[1], -x[0]))
    
    for r, c in gray_squares:
        draw_line(r, c, 1, -1, 2)  # Red line towards bottom-left
        draw_line(r, c, -1, 1, 4)  # Yellow line towards top-right
    
    return grid
