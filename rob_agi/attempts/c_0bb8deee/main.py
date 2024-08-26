from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_0bb8deee(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing horizontal and vertical lines,
    identifying important regions in corners, and scaling them to fit in a 6x6 output grid.
    
    1. Remove full-width horizontal lines.
    2. Identify the four main corner regions.
    3. Remove vertical lines not part of corner shapes.
    4. Scale and reposition corner shapes to a 6x6 output grid.
    5. Preserve relative positions and structures of shapes.
    
    Returns a new 6x6 ColoredGrid with the transformed content.
    """
    rows, cols = input_grid.get_dimensions()

    # Step 1: Remove full-width horizontal lines
    to_remove = set()
    for r in range(rows):
        if len(set(input_grid.values[r])) == 2 and 0 in input_grid.values[r]:
            to_remove.update((r, c) for c in range(cols))

    # Step 2: Identify corner regions
    def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        stack = [(r, c)]
        region = []
        visited = set()
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and (r, c) not in to_remove and 0 <= r < rows and 0 <= c < cols and input_grid.values[r][c] == color:
                visited.add((r, c))
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        return region

    quadrants = [
        (0, 0, rows // 2, cols // 2),
        (0, cols // 2, rows // 2, cols),
        (rows // 2, 0, rows, cols // 2),
        (rows // 2, cols // 2, rows, cols)
    ]

    corner_shapes = []
    for top, left, bottom, right in quadrants:
        largest_region = []
        for r in range(top, bottom):
            for c in range(left, right):
                if (r, c) not in to_remove and input_grid.values[r][c] != 0:
                    region = flood_fill(r, c, input_grid.values[r][c])
                    if len(region) > len(largest_region):
                        largest_region = region
        if largest_region:
            corner_shapes.append(largest_region)

    # Step 3: Remove vertical lines
    central_cols = [cols // 2 - 1, cols // 2] if cols % 2 == 0 else [cols // 2]
    for c in central_cols:
        vertical_line = [(r, c) for r in range(rows) if input_grid.values[r][c] != 0]
        if len(vertical_line) > rows // 2:
            to_remove.update(set(vertical_line) - set.union(*map(set, corner_shapes)))

    # Step 4 & 5: Scale and reposition shapes
    new_grid = [[0] * 6 for _ in range(6)]
    for i, shape in enumerate(corner_shapes):
        color = input_grid.values[shape[0][0]][shape[0][1]]
        min_r = min(r for r, _ in shape)
        max_r = max(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        y_scale = 3 / height
        x_scale = 3 / width
        
        quadrant_r, quadrant_c = divmod(i, 2)
        offset_r = quadrant_r * 3
        offset_c = quadrant_c * 3
        
        for r, c in shape:
            new_r = int((r - min_r) * y_scale) + offset_r
            new_c = int((c - min_c) * x_scale) + offset_c
            if 0 <= new_r < 6 and 0 <= new_c < 6:
                new_grid[new_r][new_c] = max(new_grid[new_r][new_c], color)

    return ColoredGrid(values=new_grid)
