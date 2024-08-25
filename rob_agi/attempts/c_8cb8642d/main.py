from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_8cb8642d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying rectangles, finding their seeds,
    and applying a pattern based on the rectangle's size.
    
    1. Identifies rectangles of uniform color in the grid.
    2. Finds the seed (different color pixel) within each rectangle.
    3. Applies a transformation to each rectangle:
       - For small rectangles: Creates a diamond pattern with the original color
         and fills the rest with black to form an X.
       - For large rectangles: Creates an X pattern with the original color
         and fills the rest with black to form concentric diamonds.
    4. Ensures symmetry in the transformed patterns.
    5. Applies the transformations back to the original grid.
    
    Returns the modified grid as the solution.
    """
    output_grid = input_grid.deep_copy()
    rectangles = find_rectangles(input_grid)
    
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
    seed_color, seed_pos = find_seed(grid, rect)
    
    is_small = min(height, width) <= 7  # Threshold for small rectangles
    
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if r == top or r == bottom or c == left or c == right:
                continue  # Keep the border intact
            
            rel_r, rel_c = r - top, c - left
            center_r, center_c = height // 2, width // 2
            
            if (rel_r, rel_c) in [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1), (center_r, center_c)]:
                grid.set_cell(r, c, seed_color)  # Set corners and center
            elif is_small:
                if abs(rel_r - center_r) + abs(rel_c - center_c) <= min(center_r, center_c):
                    grid.set_cell(r, c, color)  # Diamond pattern
                else:
                    grid.set_cell(r, c, 0)  # Black X
            else:
                if abs(rel_r - center_r) == abs(rel_c - center_c):
                    grid.set_cell(r, c, color)  # X pattern
                else:
                    grid.set_cell(r, c, 0)  # Black diamonds

def find_seed(grid: ColoredGrid, rect: Tuple[int, int, int, int, int]) -> Tuple[int, Tuple[int, int]]:
    """Finds the seed (different color pixel) within a rectangle."""
    top, left, bottom, right, color = rect
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if grid.get_cell(r, c) != color:
                return grid.get_cell(r, c), (r, c)
    return color, (top, left)  # Fallback if no seed found
