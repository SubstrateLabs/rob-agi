from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple, Dict
from collections import defaultdict, deque

def solve_414297c0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by selecting the most frequent non-black color as background,
    preserving all other colored elements and formations, and arranging them
    in a compact manner while maintaining their relative positions.
    
    1. Analyzes the input grid to identify the most frequent non-black color for background.
    2. Identifies all non-black elements and their formations.
    3. Creates a new grid with the chosen background color.
    4. Places preserved elements and formations, optimizing for compactness.
    5. Ensures all colored elements touch either an edge or another colored element.
    6. Optimizes the grid by removing unnecessary background-only areas.
    7. Performs final adjustments to ensure all elements are properly connected.
    
    Returns a new ColoredGrid object with the transformed grid.
    """
    # Step 1: Analyze input and select background color
    background_color = select_background_color(input_grid)
    
    # Step 2: Identify elements and formations to preserve
    color_positions = find_color_positions(input_grid, background_color)
    
    # Step 3 & 4: Create initial output grid and place elements
    output_grid = create_and_place_elements(color_positions, background_color, input_grid.get_dimensions())
    
    # Step 5 & 6: Optimize and compact the grid
    optimized_grid = optimize_and_compact_grid(output_grid, background_color)
    
    # Step 7: Perform final adjustments
    final_grid = final_adjustments(optimized_grid, background_color)
    
    # Create and return the final ColoredGrid object
    return ColoredGrid(values=final_grid)

def select_background_color(grid: ColoredGrid) -> int:
    """Selects the most frequent non-black color as the background."""
    color_counts = grid.get_color_frequencies()
    if 0 in color_counts:
        del color_counts[0]  # Remove black (0) from consideration
    return max(color_counts, key=color_counts.get) if color_counts else 0

def find_color_positions(grid: ColoredGrid, background_color: int) -> Dict[int, List[Tuple[int, int]]]:
    """Finds positions of all non-background colors."""
    color_positions = defaultdict(list)
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.values[r][c]
            if color != 0 and color != background_color:
                color_positions[color].append((r, c))
    return color_positions

def create_and_place_elements(color_positions: Dict[int, List[Tuple[int, int]]], background_color: int, original_dimensions: Tuple[int, int]) -> List[List[int]]:
    """Creates a new grid and places elements while maintaining relative positions."""
    if not color_positions:
        return [[background_color]]
    
    # Initialize with 75% of original dimensions
    rows, cols = original_dimensions
    new_rows, new_cols = max(1, int(rows * 0.75)), max(1, int(cols * 0.75))
    grid = [[background_color for _ in range(new_cols)] for _ in range(new_rows)]
    
    # Sort colors by the size of their formations (largest first)
    sorted_colors = sorted(color_positions.keys(), key=lambda c: len(color_positions[c]), reverse=True)
    
    for color in sorted_colors:
        positions = color_positions[color]
        min_r, min_c = min(positions)
        max_r, max_c = max(positions)
        height, width = max_r - min_r + 1, max_c - min_c + 1
        
        # Find a suitable position in the new grid
        placed = False
        for r in range(new_rows - height + 1):
            for c in range(new_cols - width + 1):
                if can_place_formation(grid, positions, (r, c), (min_r, min_c), background_color):
                    place_formation(grid, positions, (r, c), (min_r, min_c), color)
                    placed = True
                    break
            if placed:
                break
        
        # If can't place, expand grid and try again
        if not placed:
            grid = expand_grid(grid, background_color)
            new_rows, new_cols = len(grid), len(grid[0])
            r, c = new_rows - height, new_cols - width
            place_formation(grid, positions, (r, c), (min_r, min_c), color)
    
    return grid

def can_place_formation(grid: List[List[int]], positions: List[Tuple[int, int]], new_top_left: Tuple[int, int], old_top_left: Tuple[int, int], background_color: int) -> bool:
    new_r, new_c = new_top_left
    old_r, old_c = old_top_left
    for r, c in positions:
        nr, nc = new_r + (r - old_r), new_c + (c - old_c)
        if nr < 0 or nr >= len(grid) or nc < 0 or nc >= len(grid[0]) or grid[nr][nc] != background_color:
            return False
    return True

def place_formation(grid: List[List[int]], positions: List[Tuple[int, int]], new_top_left: Tuple[int, int], old_top_left: Tuple[int, int], color: int):
    new_r, new_c = new_top_left
    old_r, old_c = old_top_left
    for r, c in positions:
        nr, nc = new_r + (r - old_r), new_c + (c - old_c)
        grid[nr][nc] = color

def expand_grid(grid: List[List[int]], background_color: int) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    new_rows, new_cols = rows + 1, cols + 1
    new_grid = [[background_color for _ in range(new_cols)] for _ in range(new_rows)]
    for r in range(rows):
        for c in range(cols):
            new_grid[r][c] = grid[r][c]
    return new_grid

def optimize_and_compact_grid(grid: List[List[int]], background_color: int) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    
    def is_touching(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1 or any(
            grid[r + dr][c + dc] != background_color
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
        )
    
    # Compact the grid
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != background_color and not is_touching(r, c):
                    for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:  # Prioritize moving up and left
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and is_touching(nr, nc):
                            grid[nr][nc], grid[r][c] = grid[r][c], grid[nr][nc]
                            changed = True
                            break
    
    # Remove empty rows and columns
    grid = [row for row in grid if any(cell != background_color for cell in row)]
    grid = [list(col) for col in zip(*grid) if any(cell != background_color for cell in col)]
    
    return grid

def final_adjustments(grid: List[List[int]], background_color: int) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    
    def is_isolated(r: int, c: int) -> bool:
        return all(
            grid[r + dr][c + dc] == background_color
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if 0 <= r + dr < rows and 0 <= c + dc < cols
        )
    
    def find_nearest_non_background(r: int, c: int) -> Tuple[int, int]:
        queue = deque([(r, c, 0)])
        visited = set()
        while queue:
            cr, cc, dist = queue.popleft()
            if (cr, cc) not in visited:
                visited.add((cr, cc))
                if grid[cr][cc] != background_color and (cr, cc) != (r, c):
                    return cr, cc
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        queue.append((nr, nc, dist + 1))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background_color and is_isolated(r, c):
                nr, nc = find_nearest_non_background(r, c)
                # Move the isolated element next to the nearest non-background element
                if abs(nr - r) > abs(nc - c):
                    grid[nr][c], grid[r][c] = grid[r][c], grid[nr][c]
                else:
                    grid[r][nc], grid[r][c] = grid[r][c], grid[r][nc]
    
    return grid
