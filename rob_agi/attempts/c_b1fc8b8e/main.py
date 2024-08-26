from rob_agi.colored_grid import ColoredGrid

def solve_b1fc8b8e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 6x6 input grid into a 5x5 output grid based on the presence of sky blue (8) in corner regions.
    
    The function checks each 3x3 corner region in the input grid for the presence of sky blue (8).
    It then creates a 5x5 output grid where each corner is a 2x2 square that is sky blue (8) if the
    corresponding input corner contained sky blue, and black (0) otherwise. The center column and row
    are always black (0).
    
    Args:
    input_grid (ColoredGrid): A 6x6 input grid
    
    Returns:
    ColoredGrid: A 5x5 output grid with the described pattern
    """
    def check_corner_region(grid: ColoredGrid, corner: str) -> bool:
        corners = {
            'top-left': [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
            'top-right': [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)],
            'bottom-left': [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)],
            'bottom-right': [(3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
        }
        return any(grid.get_cell(r, c) == 8 for r, c in corners[corner])

    corners = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
    corner_has_sky_blue = [check_corner_region(input_grid, corner) for corner in corners]

    output_grid = [[0 for _ in range(5)] for _ in range(5)]
    output_corners = [(0, 0), (0, 3), (3, 0), (3, 3)]

    for i, (r, c) in enumerate(output_corners):
        if corner_has_sky_blue[i]:
            for dr in range(2):
                for dc in range(2):
                    output_grid[r + dr][c + dc] = 8

    return ColoredGrid(values=output_grid)
