from rob_agi.colored_grid import ColoredGrid

def solve_4ff4c9da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding sky blue (8) cells with specific rules.
    
    The function follows these steps:
    1. Analyze the input grid for patterns and structure
    2. Perform horizontal expansion within stripes
    3. Implement symmetrical mirroring
    4. Expand vertically and complete patterns
    5. Recognize and replicate distinct sky blue patterns
    6. Expand into surrounded cells
    7. Iteratively refine the expansion
    8. Perform a final pattern check
    
    Expansion rules:
    - Sky blue cells expand horizontally within their stripe (between red barriers)
    - Patterns are mirrored symmetrically across the vertical center line
    - Vertical expansion respects horizontal red lines
    - Distinct patterns are replicated in similar structural positions
    - Expansion respects the overall grid structure and symmetry
    
    Returns a new ColoredGrid with the transformed pattern.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    
    def find_stripes():
        stripes = []
        start = 0
        for c in range(cols):
            if all(new_grid.get_cell(r, c) == 2 for r in range(rows)):
                if c > start:
                    stripes.append((start, c))
                start = c + 1
        if start < cols:
            stripes.append((start, cols))
        return stripes
    
    def expand_horizontally(stripes):
        changed = False
        for start, end in stripes:
            for r in range(rows):
                if 8 in [new_grid.get_cell(r, c) for c in range(start, end)]:
                    for c in range(start, end):
                        if new_grid.get_cell(r, c) in [0, 1]:
                            new_grid.set_cell(r, c, 8)
                            changed = True
        return changed
    
    def mirror_symmetrically():
        center = cols // 2
        for r in range(rows):
            for c in range(center):
                if new_grid.get_cell(r, c) == 8:
                    mirror_c = cols - 1 - c
                    if new_grid.get_cell(r, mirror_c) in [0, 1]:
                        new_grid.set_cell(r, mirror_c, 8)
    
    def expand_vertically():
        changed = False
        for c in range(cols):
            sky_blue_rows = [r for r in range(rows) if new_grid.get_cell(r, c) == 8]
            for r in range(min(sky_blue_rows), max(sky_blue_rows) + 1):
                if new_grid.get_cell(r, c) in [0, 1]:
                    new_grid.set_cell(r, c, 8)
                    changed = True
        return changed
    
    def replicate_patterns():
        changed = False
        patterns = [
            [(0, 0), (0, 1), (0, 2)],  # Horizontal 3x1
            [(0, 0), (1, 0)],          # Vertical 2x1
            [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 square
        ]
        for r in range(rows):
            for c in range(cols):
                for pattern in patterns:
                    if all(0 <= r + dr < rows and 0 <= c + dc < cols and
                           new_grid.get_cell(r + dr, c + dc) == 8 for dr, dc in pattern):
                        for rr in range(rows):
                            for cc in range(cols):
                                if all(0 <= rr + dr < rows and 0 <= cc + dc < cols and
                                       new_grid.get_cell(rr + dr, cc + dc) in [0, 1] for dr, dc in pattern):
                                    for dr, dc in pattern:
                                        new_grid.set_cell(rr + dr, cc + dc, 8)
                                        changed = True
        return changed
    
    def expand_surrounded():
        changed = False
        for r in range(rows):
            for c in range(cols):
                if new_grid.get_cell(r, c) in [0, 1]:
                    neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= r + dr < rows and 0 <= c + dc < cols and
                                    new_grid.get_cell(r + dr, c + dc) == 8)
                    if neighbors >= 3:
                        new_grid.set_cell(r, c, 8)
                        changed = True
        return changed
    
    stripes = find_stripes()
    while True:
        changed = False
        changed |= expand_horizontally(stripes)
        mirror_symmetrically()
        changed |= expand_vertically()
        changed |= replicate_patterns()
        changed |= expand_surrounded()
        if not changed:
            break
    
    return new_grid
