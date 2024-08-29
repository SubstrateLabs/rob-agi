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
    2. Initialize a 'visited' grid to track cells that can be reached from the edges.
    3. Perform flood fill from all edge cells, including diagonal movements.
    4. Fill unvisited black cells with yellow.
    5. Return the modified grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(r, c):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < rows and 0 <= c < cols) or visited[r][c] or grid.values[r][c] in [BLUE, RED]:
                continue
            visited[r][c] = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((r + dr, c + dc))

    # Perform flood fill from all edges
    for r in [0, rows-1]:
        for c in range(cols):
            if grid.values[r][c] not in [BLUE, RED]:
                flood_fill(r, c)
    for c in [0, cols-1]:
        for r in range(rows):
            if grid.values[r][c] not in [BLUE, RED]:
                flood_fill(r, c)

    # Fill enclosed areas
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == BLACK and not visited[r][c]:
                grid.values[r][c] = YELLOW

    return grid
