from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_3490cc26(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting sky blue (8) squares with orange (7) paths.
    
    The solution follows these steps:
    1. Identify all 2x2 sky blue squares in the input grid.
    2. Determine the bounding box of all sky blue squares.
    3. Create an initial orange skeleton within the bounding box.
    4. Optimize the orange skeleton to ensure efficient connections.
    5. Connect any isolated sky blue squares.
    6. Perform final cleanup and verify connectivity.
    7. Preserve original colors for non-orange cells.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with orange paths connecting sky blue squares.
    """
    output_grid = input_grid.deep_copy()
    sky_blue_squares = find_sky_blue_squares(output_grid)
    
    if not sky_blue_squares:
        return output_grid  # No sky blue squares to connect
    
    bounding_box = get_bounding_box(sky_blue_squares)
    create_initial_skeleton(output_grid, bounding_box, sky_blue_squares)
    optimize_skeleton(output_grid, sky_blue_squares)
    connect_isolated_squares(output_grid, sky_blue_squares)
    cleanup_and_verify(output_grid, sky_blue_squares)
    
    return output_grid

def find_sky_blue_squares(grid: ColoredGrid) -> List[Tuple[int, int]]:
    """Find all 2x2 sky blue squares in the grid."""
    squares = []
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) == 8 for dr in range(2) for dc in range(2)):
                squares.append((r, c))
    return squares

def get_bounding_box(squares: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Determine the bounding box of all sky blue squares."""
    if not squares:
        return (0, 0, 0, 0)
    min_r = min(sq[0] for sq in squares)
    max_r = max(sq[0] for sq in squares)
    min_c = min(sq[1] for sq in squares)
    max_c = max(sq[1] for sq in squares)
    return (min_r, min_c, max_r + 1, max_c + 1)  # +1 to include the full 2x2 square

def create_initial_skeleton(grid: ColoredGrid, bbox: Tuple[int, int, int, int], squares: List[Tuple[int, int]]):
    """Create an initial orange skeleton within the bounding box."""
    min_r, min_c, max_r, max_c = bbox
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid.get_cell(r, c) == 0:  # Only fill black cells
                grid.set_cell(r, c, 7)  # Set to orange

def optimize_skeleton(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Optimize the orange skeleton to ensure efficient connections."""
    rows, cols = grid.get_dimensions()
    visited = set()
    
    def dfs(r, c):
        if (r, c) in visited or grid.get_cell(r, c) != 7:
            return False
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid.get_cell(nr, nc) == 8 or dfs(nr, nc):
                    return True
        grid.set_cell(r, c, 0)  # Remove unnecessary orange cells
        return False

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 7 and (r, c) not in visited:
                dfs(r, c)

def connect_isolated_squares(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Connect any isolated sky blue squares to the main orange network."""
    rows, cols = grid.get_dimensions()
    
    def bfs(start):
        queue = deque([start])
        visited = set([start])
        while queue:
            r, c = queue.popleft()
            if grid.get_cell(r, c) == 7:
                return [(r, c)]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return []

    for r, c in squares:
        if all(grid.get_cell(r+dr, c+dc) != 7 for dr in range(2) for dc in range(2)):
            path = bfs((r, c))
            for pr, pc in path:
                grid.set_cell(pr, pc, 7)

def cleanup_and_verify(grid: ColoredGrid, squares: List[Tuple[int, int]]):
    """Perform final cleanup and verify connectivity."""
    rows, cols = grid.get_dimensions()
    
    # Ensure all original sky blue squares are intact
    for r, c in squares:
        for dr in range(2):
            for dc in range(2):
                grid.set_cell(r + dr, c + dc, 8)
    
    # Remove orange cells beyond outermost sky blue squares
    bbox = get_bounding_box(squares)
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 7 and not (bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]):
                grid.set_cell(r, c, 0)
