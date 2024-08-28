from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue plus shapes to a target color.
    The target color (red or green) is determined by the color adjacent to or enclosed by gray borders.
    Only valid blue plus shapes (5 pixels in a + configuration) are transformed.
    Other blue shapes and colors remain unchanged.
    All transformations are applied simultaneously.

    1. Find gray borders and determine the target color.
    2. Identify all blue plus shapes in the grid.
    3. Transform these blue plus shapes to the target color.
    4. Return the modified grid.
    """
    GRAY, BLUE, RED, GREEN = 5, 1, 2, 3
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    result_grid = [row[:] for row in input_grid.values]

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def is_blue_plus(r: int, c: int) -> bool:
        if result_grid[r][c] != BLUE:
            return False
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if not is_valid_cell(nr, nc) or result_grid[nr][nc] != BLUE:
                return False
        return True

    gray_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == GRAY]
    
    target_color = None
    for gr, gc in gray_cells:
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = gr + dr, gc + dc
            if is_valid_cell(nr, nc) and input_grid.values[nr][nc] in [RED, GREEN]:
                target_color = input_grid.values[nr][nc]
                break
        if target_color:
            break
    
    if not target_color:
        return input_grid
    
    for r in range(rows):
        for c in range(cols):
            if is_blue_plus(r, c):
                for dr, dc in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
                    result_grid[r+dr][c+dc] = target_color
    
    return ColoredGrid(values=result_grid)
