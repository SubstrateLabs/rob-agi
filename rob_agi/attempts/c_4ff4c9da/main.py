from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells with specific rules.
    
    The function follows these steps:
    1. Create a deep copy of the input grid
    2. Perform initial expansion of sky blue cells horizontally and vertically
    3. Identify connected regions of sky blue cells
    4. Expand diagonally for each connected region
    5. Repeat steps 3-4 until no further changes occur
    
    Expansion rules:
    - Sky blue cells expand to adjacent black (0) and blue (1) cells
    - Red (2) cells and other colors act as barriers
    - Diagonal expansion only occurs if the diagonal cell is adjacent to another sky blue cell in the same region
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    def expand_horizontally_vertically():
        queue = deque((r, c) for r in range(rows) for c in range(cols) if new_grid.get_cell(r, c) == 8)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) in [0, 1]:
                    new_grid.set_cell(nr, nc, 8)
                    queue.append((nr, nc))
    
    def find_connected_regions():
        visited = set()
        regions = []
        for r in range(rows):
            for c in range(cols):
                if new_grid.get_cell(r, c) == 8 and (r, c) not in visited:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) not in visited and new_grid.get_cell(cr, cc) == 8:
                            visited.add((cr, cc))
                            region.append((cr, cc))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    stack.append((nr, nc))
                    regions.append(region)
        return regions
    
    def expand_diagonally(regions):
        changed = False
        for region in regions:
            region_set = set(region)
            for r, c in region:
                for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) in [0, 1]:
                        if any((nr + dr, nc + dc) in region_set for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]):
                            new_grid.set_cell(nr, nc, 8)
                            changed = True
        return changed
    
    expand_horizontally_vertically()
    while True:
        regions = find_connected_regions()
        if not expand_diagonally(regions):
            break
    
    return new_grid
