from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e2092e0c(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a continuous path of 5's (gray).
    
    The solution follows these steps:
    1. Find a starting point (existing 5 near edge or corner, or a corner cell).
    2. Create a path along the grid edge, then turn inward.
    3. Extend the path to form a significant shape (L-shape or similar).
    4. Ensure the path is at least 10-15 cells long.
    5. Update the grid with the new path of 5's.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with a new continuous path of 5's.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def find_start_point() -> Tuple[int, int]:
        # Check for existing 5's near edges
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 5 and (r in [0, rows-1] or c in [0, cols-1]):
                    return r, c
        # If no suitable 5 found, start from top-left corner
        return 0, 0
    
    def generate_path(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [start]
        r, c = start
        
        # Main segment along edge
        while len(path) < 5:
            if c < cols - 1:
                c += 1
            elif r < rows - 1:
                r += 1
            path.append((r, c))
        
        # Turn inward
        if r == 0:
            r += 1
        elif c == cols - 1:
            c -= 1
        elif r == rows - 1:
            r -= 1
        else:
            c += 1
        path.append((r, c))
        
        # Secondary segment
        direction = (1, 0) if r < rows // 2 else (-1, 0)
        while len(path) < 10:
            r += direction[0]
            c += direction[1]
            if 0 <= r < rows and 0 <= c < cols:
                path.append((r, c))
            else:
                break
        
        return path
    
    start_point = find_start_point()
    path = generate_path(start_point)
    
    # Update grid with new path
    for r, c in path:
        output_grid.set_cell(r, c, 5)
    
    return output_grid
