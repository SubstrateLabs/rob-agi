from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing blue plus shapes to a target color.
    The target color (red or green) is determined by the color adjacent to gray borders.
    Only valid blue plus shapes (5 pixels in a + configuration) are transformed.
    Other blue shapes and colors remain unchanged.
    All transformations are applied simultaneously.

    1. Find the target color adjacent to gray borders.
    2. Identify all blue plus shapes in the grid.
    3. Transform these blue plus shapes to the target color.
    4. Return the modified grid.
    """
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    result_grid = [row[:] for row in input_grid.values]
    target_color = None

    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def is_plus_shape(r: int, c: int) -> bool:
        if result_grid[r][c] != BLUE:
            return False
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if not is_valid_cell(nr, nc) or result_grid[nr][nc] != BLUE:
                return False
        return True

    # Find the target color and transform blue plus shapes in a single pass
    for i in range(rows):
        for j in range(cols):
            if result_grid[i][j] == GRAY and target_color is None:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if is_valid_cell(ni, nj) and result_grid[ni][nj] in [RED, GREEN]:
                        target_color = result_grid[ni][nj]
                        break
                if target_color:
                    break
            elif is_plus_shape(i, j) and target_color is not None:
                for di, dj in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
                    result_grid[i+di][j+dj] = target_color

    return ColoredGrid(values=result_grid)
