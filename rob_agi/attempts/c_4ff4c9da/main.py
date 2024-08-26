from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells with specific rules.
    
    The function follows these steps:
    1. Create a deep copy of the input grid
    2. Perform initial 3x3 expansion around sky blue cells
    3. Expand vertical and horizontal lines of sky blue cells
    4. Identify connected regions of sky blue cells
    5. Expand complex regions based on specific patterns
    6. Repeat steps 2-5 until no further changes occur
    
    Expansion rules:
    - Sky blue cells expand to adjacent black (0) and blue (1) cells
    - Red (2) cells and other colors act as barriers
    - Specific patterns trigger special expansion rules
    - Expansion respects grid boundaries and existing patterns
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    def expand_3x3():
        changed = False
        for r in range(rows):
            for c in range(cols):
                if new_grid.get_cell(r, c) == 8:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) in [0, 1]:
                                new_grid.set_cell(nr, nc, 8)
                                changed = True
        return changed
    
    def expand_lines():
        changed = False
        for r in range(rows):
            for c in range(cols):
                if new_grid.get_cell(r, c) == 8:
                    # Expand horizontally
                    for dc in [-1, 1]:
                        nc = c + dc
                        while 0 <= nc < cols and new_grid.get_cell(r, nc) in [0, 1]:
                            new_grid.set_cell(r, nc, 8)
                            changed = True
                            nc += dc
                    # Expand vertically
                    for dr in [-1, 1]:
                        nr = r + dr
                        while 0 <= nr < rows and new_grid.get_cell(nr, c) in [0, 1]:
                            new_grid.set_cell(nr, c, 8)
                            changed = True
                            nr += dr
        return changed
    
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
    
    def expand_complex_regions(regions):
        changed = False
        for region in regions:
            region_set = set(region)
            for r, c in region:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) in [0, 1]:
                        adjacent_sky_blue = sum(1 for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)] if (nr + dr2, nc + dc2) in region_set)
                        if adjacent_sky_blue >= 2:
                            new_grid.set_cell(nr, nc, 8)
                            changed = True
        return changed
    
    while True:
        changed = False
        changed |= expand_3x3()
        changed |= expand_lines()
        regions = find_connected_regions()
        changed |= expand_complex_regions(regions)
        if not changed:
            break
    
    return new_grid
