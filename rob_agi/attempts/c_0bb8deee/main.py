from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_0bb8deee(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing horizontal and vertical lines,
    identifying important shapes in quadrants, and fitting them into a 6x6 output grid.
    
    1. Remove full-width horizontal and full-height vertical lines.
    2. Divide the cleaned grid into four quadrants.
    3. Identify significant shapes in each quadrant using flood fill.
    4. Scale and reposition shapes to fit in a 6x6 output grid.
    5. Preserve relative positions and structures of shapes.
    
    Returns a new 6x6 ColoredGrid with the transformed content.
    """
    rows, cols = input_grid.get_dimensions()

    # Step 1: Remove full-width horizontal and full-height vertical lines
    to_remove = set()
    for r in range(rows):
        if len(set(input_grid.values[r])) == 1 and input_grid.values[r][0] != 0:
            to_remove.update((r, c) for c in range(cols))
    for c in range(cols):
        if len(set(input_grid.values[r][c] for r in range(rows))) == 1 and input_grid.values[0][c] != 0:
            to_remove.update((r, c) for r in range(rows))

    # Step 2: Divide the cleaned grid into four quadrants
    mid_row, mid_col = rows // 2, cols // 2
    quadrants = [
        (0, 0, mid_row, mid_col),
        (0, mid_col, mid_row, cols),
        (mid_row, 0, rows, mid_col),
        (mid_row, mid_col, rows, cols)
    ]

    # Step 3: Identify significant shapes in each quadrant
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

    quadrant_shapes = []
    for top, left, bottom, right in quadrants:
        shapes = []
        for r in range(top, bottom):
            for c in range(left, right):
                if (r, c) not in to_remove and input_grid.values[r][c] != 0:
                    region = flood_fill(r, c, input_grid.values[r][c])
                    if region:
                        shapes.append(region)
        quadrant_shapes.append(sorted(shapes, key=len, reverse=True))

    # Step 4 & 5: Scale and reposition shapes
    new_grid = [[0] * 6 for _ in range(6)]
    for i, shapes in enumerate(quadrant_shapes):
        quadrant_r, quadrant_c = divmod(i, 2)
        offset_r, offset_c = quadrant_r * 3, quadrant_c * 3
        
        for shape in shapes[:3]:  # Consider up to 3 largest shapes per quadrant
            color = input_grid.values[shape[0][0]][shape[0][1]]
            min_r = min(r for r, _ in shape)
            max_r = max(r for r, _ in shape)
            min_c = min(c for _, c in shape)
            max_c = max(c for _, c in shape)
            
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            y_scale = min(3 / height, 1)
            x_scale = min(3 / width, 1)
            
            for r, c in shape:
                new_r = int((r - min_r) * y_scale) + offset_r
                new_c = int((c - min_c) * x_scale) + offset_c
                if 0 <= new_r < offset_r + 3 and 0 <= new_c < offset_c + 3:
                    new_grid[new_r][new_c] = color

    return ColoredGrid(values=new_grid)
