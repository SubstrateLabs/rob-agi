from rob_agi.colored_grid import ColoredGrid

BLACK, BLUE, RED, YELLOW = 0, 1, 2, 4

def solve_b15fca0b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed areas with yellow (4).
    
    The function identifies areas that are completely surrounded by blue (1) lines,
    red (2) squares, or the grid edges. These enclosed areas are filled with yellow (4).
    Areas that have a path to any edge of the grid (including diagonal paths) remain black (0).
    Blue lines and red squares remain unchanged.
    
    Algorithm:
    1. Create a deep copy of the input grid.
    2. Initialize an 'open' grid to track cells that can be reached from any edge.
    3. Perform flood fill from all edge cells that are not blue or red, including diagonal movements.
    4. Fill unreachable black cells with yellow.
    5. Return the modified grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    open_grid = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(r, c):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < rows and 0 <= c < cols) or open_grid[r][c] or grid.values[r][c] in [BLUE, RED]:
                continue
            open_grid[r][c] = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    stack.append((r + dr, c + dc))

    # Perform flood fill from all edges
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                flood_fill(r, c)

    # Fill enclosed areas
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == BLACK and not open_grid[r][c]:
                grid.values[r][c] = YELLOW

    return grid
