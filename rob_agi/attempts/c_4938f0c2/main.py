from rob_agi.colored_grid import ColoredGrid
import copy

def solve_4938f0c2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the puzzle by mirroring the red (2) cells around a central green (3) block.
    
    1. Finds the central 2x2 block of green (3) cells.
    2. For each red (2) cell, mirrors it horizontally, vertically, and diagonally
       around the center block.
    3. Returns the modified grid with mirrored red cells.

    If no central green block is found, returns the original grid unchanged.
    """
    def find_center_block(grid):
        rows, cols = len(grid), len(grid[0])
        for i in range(rows - 1):
            for j in range(cols - 1):
                if all(grid[i+di][j+dj] == 3 for di in range(2) for dj in range(2)):
                    return (i, j)
        return None

    def mirror_grid(grid, center):
        rows, cols = len(grid), len(grid[0])
        mirrored_grid = copy.deepcopy(grid)
        center_row, center_col = center

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    # Horizontal mirroring
                    mirrored_col = 2 * center_col + 1 - j
                    if 0 <= mirrored_col < cols:
                        mirrored_grid[i][mirrored_col] = 2
                    
                    # Vertical mirroring
                    mirrored_row = 2 * center_row + 1 - i
                    if 0 <= mirrored_row < rows:
                        mirrored_grid[mirrored_row][j] = 2
                    
                    # Both horizontal and vertical mirroring
                    if 0 <= mirrored_col < cols and 0 <= mirrored_row < rows:
                        mirrored_grid[mirrored_row][mirrored_col] = 2

        return mirrored_grid

    grid = input_grid.values
    center = find_center_block(grid)

    if center is None:
        return input_grid  # No central block found, return the original grid

    mirrored_grid = mirror_grid(grid, center)
    return ColoredGrid(values=mirrored_grid)
