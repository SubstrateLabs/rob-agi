from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_0bb8deee(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing full-width lines, identifying important shapes,
    and fitting them into a 6x6 output grid while preserving relative positions.
    
    1. Remove full-width horizontal and full-height vertical lines.
    2. Identify significant shapes using flood fill.
    3. Determine relative positions of shapes.
    4. Scale and reposition shapes to fit in a 6x6 output grid.
    5. Adjust for overlaps and preserve relative structure.
    
    Returns a new 6x6 ColoredGrid with the transformed content.
    """
    rows, cols = input_grid.get_dimensions()

    # Step 1: Remove full-width lines
    to_remove = set()
    for r in range(rows):
        if len(set(input_grid.values[r])) == 1 and input_grid.values[r][0] != 0:
            to_remove.update((r, c) for c in range(cols))
    for c in range(cols):
        if len(set(input_grid.values[r][c] for r in range(rows))) == 1 and input_grid.values[0][c] != 0:
            to_remove.update((r, c) for r in range(rows))

    # Step 2: Identify shapes
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

    shapes = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in to_remove and input_grid.values[r][c] != 0:
                region = flood_fill(r, c, input_grid.values[r][c])
                if region:
                    shapes.append(region)

    # Step 3: Determine relative positions
    center_r, center_c = rows // 2, cols // 2
    for shape in shapes:
        avg_r = sum(r for r, _ in shape) / len(shape)
        avg_c = sum(c for _, c in shape) / len(shape)
        shape_position = (avg_r < center_r, avg_c < center_c)
        shape.append(shape_position)

    # Step 4 & 5: Scale and reposition shapes
    new_grid = [[0] * 6 for _ in range(6)]
    for shape in sorted(shapes, key=len, reverse=True):
        color = input_grid.values[shape[0][0]][shape[0][1]]
        position = shape[-1]
        shape = shape[:-1]  # Remove position tuple
        
        min_r, max_r = min(r for r, _ in shape), max(r for r, _ in shape)
        min_c, max_c = min(c for _, c in shape), max(c for _, c in shape)
        height, width = max_r - min_r + 1, max_c - min_c + 1
        
        quadrant_r = 0 if position[0] else 3
        quadrant_c = 0 if position[1] else 3
        
        scale = min(3 / max(height, width), 1)
        
        for r, c in shape:
            new_r = int((r - min_r) * scale) + quadrant_r
            new_c = int((c - min_c) * scale) + quadrant_c
            if 0 <= new_r < 6 and 0 <= new_c < 6:
                new_grid[new_r][new_c] = color

    return ColoredGrid(values=new_grid)
