from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_cb227835(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by creating a rectangular path between two sky-colored squares.
    
    The function finds the two sky-colored (8) squares in the input grid,
    creates a rectangular path between them, and marks the path with green (3) squares.
    The path forms a rectangle with the sky squares at opposite corners.
    Three sides of the rectangle are filled based on the relative positions of the sky squares.
    
    The original sky squares remain unchanged.
    
    Args:
    input_grid (ColoredGrid): The input grid containing two sky-colored squares.
    
    Returns:
    ColoredGrid: A new grid with the rectangular path marked in green.
    """
    def find_sky_squares(grid: List[List[int]]) -> List[Tuple[int, int]]:
        return [(c, r) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 8]
    
    def create_rectangular_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        x1, y1 = start
        x2, y2 = end
        path = []
        
        if x1 == x2 or y1 == y2:  # Straight line
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    path.append((x, y))
        else:
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            if width > height:
                # Fill top, bottom, and left sides
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    path.append((x, y1))
                    path.append((x, y2))
                for y in range(min(y1, y2) + 1, max(y1, y2)):
                    path.append((min(x1, x2), y))
            else:
                # Fill left, right, and bottom sides
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    path.append((x1, y))
                    path.append((x2, y))
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    path.append((x, max(y1, y2)))
        
        return path
    
    sky_squares = find_sky_squares(input_grid.values)
    if len(sky_squares) != 2:
        return input_grid  # Return original grid if there aren't exactly two sky squares
    
    start, end = sky_squares
    path = create_rectangular_path(start, end)
    
    new_grid = input_grid.deep_copy()
    for x, y in path:
        if new_grid.values[y][x] != 8:  # Don't overwrite sky squares
            new_grid.values[y][x] = 3  # Set to green
    
    return new_grid
