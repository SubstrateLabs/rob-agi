from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_8cb8642d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying rectangles, finding their seeds,
    and applying a pattern based on the rectangle's size and seed color.
    
    1. Identifies rectangles of uniform color in the grid.
    2. Finds the seed (different color pixel) within each rectangle.
    3. Generates a pattern for each rectangle:
       - Creates an X pattern with the seed color.
       - Fills the corners and center with the seed color.
       - Adds additional lines based on rectangle size.
       - Fills the rest with black (0).
    4. Applies the generated patterns to the original grid.
    5. Preserves the original border of each rectangle.
    
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
    
    pattern = generate_pattern(width - 2, height - 2, seed_color)
    
    # Apply pattern to grid
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            grid.set_cell(r, c, pattern[r - top - 1][c - left - 1])

def generate_pattern(width: int, height: int, seed_color: int) -> List[List[int]]:
    """Generates a pattern for the given dimensions and seed color."""
    pattern = [[0 for _ in range(width)] for _ in range(height)]
    
    # Helper function to set a cell if it's within bounds
    def set_cell(r: int, c: int, color: int):
        if 0 <= r < height and 0 <= c < width:
            pattern[r][c] = color
    
    # Set corners and center
    set_cell(0, 0, seed_color)
    set_cell(0, width - 1, seed_color)
    set_cell(height - 1, 0, seed_color)
    set_cell(height - 1, width - 1, seed_color)
    set_cell(height // 2, width // 2, seed_color)
    
    # Draw main diagonals
    for i in range(min(width, height)):
        set_cell(i, i, seed_color)
        set_cell(i, width - 1 - i, seed_color)
    
    # Draw additional lines based on size
    num_lines = min(width, height) // 2
    for i in range(1, num_lines):
        # Horizontal lines
        for c in range(width):
            set_cell(i, c, seed_color)
            set_cell(height - 1 - i, c, seed_color)
        # Vertical lines
        for r in range(height):
            set_cell(r, i, seed_color)
            set_cell(r, width - 1 - i, seed_color)
    
    return pattern

def find_seed(grid: ColoredGrid, rect: Tuple[int, int, int, int, int]) -> Tuple[int, Tuple[int, int]]:
    """Finds the seed (different color pixel) within a rectangle."""
    top, left, bottom, right, color = rect
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if grid.get_cell(r, c) != color:
                return grid.get_cell(r, c), (r, c)
    return color, (top, left)  # Fallback if no seed found
