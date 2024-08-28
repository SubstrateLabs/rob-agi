from rob_agi.colored_grid import ColoredGrid

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid based on the following rules:
    1. Blue plus shapes (5 pixels) are changed to the target color (red or green).
    2. The target color (red or green) is determined by the color enclosed in a gray border.
    3. Other blue shapes and colors remain unchanged.
    4. Gray (5) acts as a border and is not considered part of any region.
    5. All transformations are applied simultaneously.

    The solution scans for a gray-enclosed target color, identifies blue plus shapes,
    and transforms them to the target color in a single pass.
    """
    BLUE, RED, GREEN, GRAY = 1, 2, 3, 5
    rows, cols = len(input_grid.values), len(input_grid.values[0])
    result_grid = [row[:] for row in input_grid.values]
    target_color = None

    # Find the target color
    for i in range(rows):
        for j in range(cols):
            if input_grid.values[i][j] == GRAY:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if input_grid.values[ni][nj] in [RED, GREEN]:
                            target_color = input_grid.values[ni][nj]
                            break
                if target_color:
                    break
        if target_color:
            break

    if target_color is None:
        return input_grid  # No transformation needed

    def is_plus_shape(x: int, y: int) -> bool:
        if input_grid.values[x][y] != BLUE:
            return False
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < rows and 0 <= ny < cols and input_grid.values[nx][ny] == BLUE):
                return False
        return True

    # Transform blue plus shapes
    for i in range(rows):
        for j in range(cols):
            if is_plus_shape(i, j):
                for di, dj in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
                    result_grid[i+di][j+dj] = target_color

    return ColoredGrid(values=result_grid)
