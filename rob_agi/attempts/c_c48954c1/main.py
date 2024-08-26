from rob_agi.colored_grid import ColoredGrid

def solve_c48954c1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 grid by applying rotations and reflections.
    
    The input grid is placed in the center of the 9x9 grid. The surrounding sections
    are filled with transformed versions of the input:
    - Center: Original input
    - All corners (Top-left, Top-right, Bottom-left, Bottom-right): Rotated 180°
    - Top-center, Bottom-center: Reflected horizontally
    - Middle-left, Middle-right: Reflected vertically
    """
    def rotate_180(grid):
        return [row[::-1] for row in grid[::-1]]

    def mirror_horizontal(grid):
        return [row[::-1] for row in grid]

    def mirror_vertical(grid):
        return grid[::-1]

    new_grid = [[0 for _ in range(9)] for _ in range(9)]

    # Place the original input in the center
    for i in range(3):
        for j in range(3):
            new_grid[i+3][j+3] = input_grid.values[i][j]

    # Fill corner quadrants (all with 180° rotation)
    rotated = rotate_180(input_grid.values)
    for i in range(3):
        new_grid[i][:3] = rotated[i]
        new_grid[i][6:] = rotated[i]
        new_grid[i+6][:3] = rotated[i]
        new_grid[i+6][6:] = rotated[i]

    # Fill edge quadrants
    horizontal_mirror = mirror_horizontal(input_grid.values)
    vertical_mirror = mirror_vertical(input_grid.values)

    for i in range(3):
        new_grid[i][3:6] = horizontal_mirror[i]
        new_grid[i+6][3:6] = horizontal_mirror[i]
        new_grid[i+3][:3] = vertical_mirror[i]
        new_grid[i+3][6:] = vertical_mirror[i]

    return ColoredGrid(values=new_grid)
