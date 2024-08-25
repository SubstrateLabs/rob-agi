from rob_agi.colored_grid import ColoredGrid
import copy

def create_propagation_plan(grid, start_row, start_col, color):
    plan = {(start_row, start_col): 0}
    rows, cols = len(grid), len(grid[0])
    
    def propagate(row, col, distance):
        if distance > 4 or row < 0 or row >= rows or col < 0 or col >= cols:
            return
        if grid[row][col] not in [0, 5] and (row, col) != (start_row, start_col):
            return
        if (row, col) not in plan or plan[(row, col)] > distance:
            plan[(row, col)] = distance
        new_distance = 0 if grid[row][col] == 5 else distance + 1
        if color == 2:  # Red
            for dr, dc in [(-1, -1), (0, -1), (1, -1)]:
                propagate(row + dr, col + dc, new_distance)
        else:  # Yellow
            for dr, dc in [(-1, 1), (0, 1), (1, 1)]:
                propagate(row + dr, col + dc, new_distance)
    
    propagate(start_row, start_col, 0)
    return plan

def apply_propagation_plan(grid, plan, color):
    for (row, col), distance in plan.items():
        if grid[row][col] == 0 or (color == 4 and grid[row][col] == 2):
            grid[row][col] = color

def solve_212895b5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by propagating colors from gray squares.
    
    For each gray (5) square, starting from bottom-right to top-left:
    1. Create a propagation plan for red (2) color towards bottom-left.
    2. Create a propagation plan for yellow (4) color towards top-right.
    3. Apply the red propagation plan, then the yellow propagation plan.
    4. Limit propagation to 4 steps from the origin, resetting at gray squares.
    5. Stop propagation at grid edges or when encountering pre-existing colors.
    6. Yellow overwrites red, but not vice versa.
    7. Preserve original gray squares and other pre-existing colors.
    
    Args:
    input_grid (ColoredGrid): The input grid to transform.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = copy.deepcopy(input_grid.values)
    gray_squares = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 5]
    gray_squares.sort(key=lambda x: (-x[0] - x[1], -x[0]))
    
    for row, col in gray_squares:
        red_plan = create_propagation_plan(grid, row, col, 2)
        yellow_plan = create_propagation_plan(grid, row, col, 4)
        apply_propagation_plan(grid, red_plan, 2)
        apply_propagation_plan(grid, yellow_plan, 4)
    
    for row, col in gray_squares:
        grid[row][col] = 5
    
    return ColoredGrid(values=grid)
