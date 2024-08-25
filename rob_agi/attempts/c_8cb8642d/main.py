from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_8cb8642d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying rectangles, finding their seeds,
    and applying an X-pattern based on the rectangle's size.
    
    1. Identifies rectangles of uniform color in the grid.
    2. Finds the seed (different color pixel) within each rectangle.
    3. Applies a transformation to each rectangle:
       - Creates an X pattern with the seed color.
       - Fills the corners and center with the seed color.
       - Fills the rest with black (0).
    4. Ensures symmetry in the transformed patterns.
    5. Applies the transformations back to the original grid.
    6. Preserves the original border of each rectangle.
    
    Returns the modified grid as the solution.
    """
    output_grid = input_grid.deep_copy()
    rectangles = find_rectangles(output_grid)
    
    for rect in rectangles:
        transform_rectangle(output_grid, rect)
    
    return output_grid

def find_rectangles(grid: ColoredGrid) -> List[Tuple[int, int, int, int, int]]:
    """Finds rectangles in the grid and returns their coordinates and color."""
    rectangles = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid.get_cell(r, c) != 0:
                color = grid.get_cell(r, c)
                top, left, bottom, right = r, c, r, c
                
                # Expand rectangle
                while right + 1 < cols and grid.get_cell(r, right + 1) == color:
                    right += 1
                while bottom + 1 < rows and all(grid.get_cell(bottom + 1, cc) == color for cc in range(left, right + 1)):
                    bottom += 1
                
                # Mark as visited
                for rr in range(top, bottom + 1):
                    for cc in range(left, right + 1):
                        visited.add((rr, cc))
                
                rectangles.append((top, left, bottom, right, color))
    
    return rectangles

def transform_rectangle(grid: ColoredGrid, rect: Tuple[int, int, int, int, int]):
    """Applies the transformation to a single rectangle."""
    top, left, bottom, right, color = rect
    height, width = bottom - top + 1, right - left + 1
    seed_color, _ = find_seed(grid, rect)
    
    if seed_color == color or height <= 2 or width <= 2:
        return  # No transformation needed if no seed found or rectangle is too small
    
    # Apply pattern to grid
    for r in range(top+1, bottom):
        for c in range(left+1, right):
            x = (c - left - 1) / (width - 3)
            y = (r - top - 1) / (height - 3)
            if abs(x - y) < 0.1 or abs(x - (1-y)) < 0.1 or abs(x - 0.5) < 0.1 or abs(y - 0.5) < 0.1:
                grid.set_cell(r, c, seed_color)
            else:
                grid.set_cell(r, c, 0)

def find_seed(grid: ColoredGrid, rect: Tuple[int, int, int, int, int]) -> Tuple[int, Tuple[int, int]]:
    """Finds the seed (different color pixel) within a rectangle."""
    top, left, bottom, right, color = rect
    for r in range(top+1, bottom):
        for c in range(left+1, right):
            if grid.get_cell(r, c) != color:
                return grid.get_cell(r, c), (r, c)
    return color, (top, left)  # Fallback if no seed found
