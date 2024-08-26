from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0e671a1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting colored squares with a gray path.
    
    1. Find the three colored squares (red, yellow, green).
    2. Determine the bounding box of these squares, extending to grid edges when possible.
    3. Create a clockwise path around the bounding box, including the colored squares.
    4. Fill the enclosed area with gray, preserving the original colored squares.
    
    Args:
    input_grid (ColoredGrid): The input grid with three colored squares.
    
    Returns:
    ColoredGrid: The transformed grid with the gray path connecting the colored squares.
    """
    def find_colored_squares(grid: ColoredGrid) -> List[Tuple[int, int, int]]:
        squares = []
        for r in range(grid.num_rows):
            for c in range(grid.num_cols):
                if grid[r][c] in [2, 3, 4]:
                    squares.append((r, c, grid[r][c]))
        return squares

    def get_bounding_box(squares: List[Tuple[int, int, int]], rows: int, cols: int) -> Tuple[int, int, int, int]:
        min_r = min(s[0] for s in squares)
        max_r = max(s[0] for s in squares)
        min_c = min(s[1] for s in squares)
        max_c = max(s[1] for s in squares)
        return (0, 0, rows - 1, cols - 1)  # Extend to grid edges

    def create_path(grid: ColoredGrid, box: Tuple[int, int, int, int], squares: List[Tuple[int, int, int]]):
        min_r, min_c, max_r, max_c = box
        for r in range(min_r, max_r + 1):
            grid[r][min_c] = 5
            grid[r][max_c] = 5
        for c in range(min_c, max_c + 1):
            grid[min_r][c] = 5
            grid[max_r][c] = 5
        for r, c, color in squares:
            grid[r][c] = color  # Restore original colors

    def flood_fill(grid: ColoredGrid, r: int, c: int):
        if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid[r][c] == 0:
            grid[r][c] = 5
            flood_fill(grid, r+1, c)
            flood_fill(grid, r-1, c)
            flood_fill(grid, r, c+1)
            flood_fill(grid, r, c-1)

    output_grid = input_grid.deep_copy()
    colored_squares = find_colored_squares(output_grid)
    bounding_box = get_bounding_box(colored_squares, output_grid.num_rows, output_grid.num_cols)
    create_path(output_grid, bounding_box, colored_squares)
    
    # Fill the enclosed area
    min_r, min_c, max_r, max_c = bounding_box
    flood_fill(output_grid, min_r + 1, min_c + 1)

    return output_grid
