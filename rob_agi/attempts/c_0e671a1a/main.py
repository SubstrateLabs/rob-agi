from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple
import math

def solve_0e671a1a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting colored squares with a gray path.
    
    1. Find the three colored squares (red, yellow, green).
    2. Sort the squares in clockwise order.
    3. Create a clockwise path connecting all squares.
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

    def calculate_center(squares: List[Tuple[int, int, int]]) -> Tuple[float, float]:
        return sum(s[0] for s in squares) / 3, sum(s[1] for s in squares) / 3

    def calculate_angle(point: Tuple[int, int], center: Tuple[float, float]) -> float:
        return math.atan2(point[0] - center[0], point[1] - center[1])

    def sort_squares_clockwise(squares: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        center = calculate_center(squares)
        return sorted(squares, key=lambda s: calculate_angle(s, center), reverse=True)

    def create_path(grid: ColoredGrid, squares: List[Tuple[int, int, int]]):
        path = set()
        for i in range(len(squares)):
            start = squares[i]
            end = squares[(i + 1) % len(squares)]
            
            # Horizontal movement
            for c in range(min(start[1], end[1]), max(start[1], end[1]) + 1):
                path.add((start[0], c))
            
            # Vertical movement
            for r in range(min(start[0], end[0]), max(start[0], end[0]) + 1):
                path.add((r, end[1]))
        
        # Draw the path
        for r, c in path:
            if grid[r][c] == 0:  # Only fill if it's an empty cell
                grid[r][c] = 5
        return path

    def flood_fill(grid: ColoredGrid, r: int, c: int, path: set):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < grid.num_rows and 0 <= c < grid.num_cols and grid[r][c] == 0 and (r, c) not in path:
                grid[r][c] = 5
                stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

    output_grid = input_grid.deep_copy()
    colored_squares = find_colored_squares(output_grid)
    sorted_squares = sort_squares_clockwise(colored_squares)
    path = create_path(output_grid, sorted_squares)
    
    # Find a starting point for flood fill (just inside the path)
    center = calculate_center(sorted_squares)
    center_r, center_c = int(center[0]), int(center[1])
    flood_fill(output_grid, center_r, center_c, path)

    # Restore original colored squares
    for r, c, color in colored_squares:
        output_grid[r][c] = color

    return output_grid
