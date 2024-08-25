from rob_agi.colored_grid import ColoredGrid

def solve_b15fca0b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling enclosed areas with yellow (4).
    
    The function identifies areas that are completely surrounded by blue (1) lines,
    red (2) squares, or the grid edges. These enclosed areas are filled with yellow (4).
    Areas that have a path to the edge of the grid remain black (0).
    Blue lines and red squares remain unchanged.
    
    Algorithm:
    1. Use a flood fill algorithm to mark all cells that have a path to the edge.
    2. Convert unmarked cells (enclosed areas) to yellow (4).
    3. Restore the original colors of blue lines and red squares.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()

    def flood_fill(r, c):
        if not (0 <= r < rows and 0 <= c < cols):
            return True  # Reached the edge
        if new_grid.values[r][c] in [1, 2, -1]:  # Blue, red, or already visited
            return False
        if new_grid.values[r][c] == 0:
            new_grid.values[r][c] = -1  # Mark as visited
            # Check all four directions
            return (flood_fill(r-1, c) or flood_fill(r+1, c) or
                    flood_fill(r, c-1) or flood_fill(r, c+1))
        return False

    for r in range(rows):
        for c in range(cols):
            if new_grid.values[r][c] == 0:
                if not flood_fill(r, c):
                    new_grid.values[r][c] = 0  # Change back to black if no path to edge

    for r in range(rows):
        for c in range(cols):
            if new_grid.values[r][c] == 0:
                new_grid.values[r][c] = 4  # Change to yellow
            elif new_grid.values[r][c] == -1:
                new_grid.values[r][c] = 0  # Change back to black

    return new_grid
