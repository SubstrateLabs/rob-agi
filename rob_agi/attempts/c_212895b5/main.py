from rob_agi.colored_grid import ColoredGrid
import copy
from collections import deque

def solve_212895b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating colors diagonally from gray squares.
    
    For each gray (5) square, starting from bottom-right to top-left:
    1. Propagate red (2) color diagonally towards bottom-left.
    2. Propagate yellow (4) color diagonally towards top-right.
    3. Continue propagation through gray squares and from newly colored squares.
    4. Stop propagation at grid edges or when encountering colors from earlier propagations.
    5. Resolve color conflicts based on the order of original gray squares.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = copy.deepcopy(input_grid)
    rows, cols = len(grid.values), len(grid.values[0])
    
    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols
    
    def get_next_positions(r, c, direction):
        if direction == 'down-left':
            return [(r+1, c-1), (r+1, c), (r, c-1)]
        else:  # 'up-right'
            return [(r-1, c+1), (r-1, c), (r, c+1)]
    
    gray_squares = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 5]
    gray_squares.sort(key=lambda x: (-x[0] - x[1], -x[0]))
    
    color_origins = {}  # To track the origin of each colored square
    
    propagation_queue = deque()
    for index, (r, c) in enumerate(gray_squares):
        propagation_queue.append((r, c, 2, 'down-left', index))  # Red propagation
        propagation_queue.append((r, c, 4, 'up-right', index))   # Yellow propagation
    
    while propagation_queue:
        r, c, color, direction, origin_index = propagation_queue.popleft()
        if not is_valid(r, c):
            continue
        
        current_color = grid.values[r][c]
        if current_color not in [0, 5]:
            if origin_index > color_origins.get((r, c), -1):
                grid.values[r][c] = color
                color_origins[(r, c)] = origin_index
            continue
        
        if current_color == 0:
            grid.values[r][c] = color
            color_origins[(r, c)] = origin_index
        
        for next_r, next_c in get_next_positions(r, c, direction):
            propagation_queue.append((next_r, next_c, color, direction, origin_index))
    
    return grid
