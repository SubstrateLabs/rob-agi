from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_0bb8deee(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by removing horizontal and vertical lines,
    identifying important regions, and scaling them to fit in a 6x6 output grid.
    
    1. Remove horizontal lines that span the entire width.
    2. Remove vertical lines in the central column(s).
    3. Identify important regions using flood fill.
    4. Scale and transfer preserved shapes to a 6x6 output grid.
    
    Returns a new 6x6 ColoredGrid with the transformed content.
    """
    rows, cols = input_grid.get_dimensions()
    central_col = cols // 2

    # Step 1 & 2: Remove horizontal and vertical lines
    to_remove = set()
    for r in range(rows):
        if len(set(input_grid.values[r])) == 2 and 0 in input_grid.values[r]:
            to_remove.update((r, c) for c in range(cols))
    
    vertical_line = set()
    for r in range(rows):
        if input_grid.values[r][central_col] != 0:
            vertical_line.add((r, central_col))
        else:
            if len(vertical_line) > 1:
                to_remove.update(vertical_line)
            vertical_line.clear()
    if len(vertical_line) > 1:
        to_remove.update(vertical_line)

    # Step 3: Identify important regions
    regions = []
    visited = set()

    def flood_fill(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        stack = [(r, c)]
        region = []
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and (r, c) not in to_remove and 0 <= r < rows and 0 <= c < cols and input_grid.values[r][c] == color:
                visited.add((r, c))
                region.append((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((r + dr, c + dc))
        return region

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and (r, c) not in to_remove and input_grid.values[r][c] != 0:
                region = flood_fill(r, c, input_grid.values[r][c])
                if region:
                    regions.append(region)

    # Step 4 & 5: Prepare for scaling
    if not regions:
        return ColoredGrid(values=[[0] * 6 for _ in range(6)])

    min_r = min(r for region in regions for r, _ in region)
    max_r = max(r for region in regions for r, _ in region)
    min_c = min(c for region in regions for _, c in region)
    max_c = max(c for region in regions for _, c in region)

    y_scale = 6 / (max_r - min_r + 1)
    x_scale = 6 / (max_c - min_c + 1)

    # Step 6 & 7: Create and fill the new 6x6 grid
    new_grid = [[0] * 6 for _ in range(6)]
    for region in regions:
        color = input_grid.values[region[0][0]][region[0][1]]
        for r, c in region:
            new_r = int((r - min_r) * y_scale)
            new_c = int((c - min_c) * x_scale)
            new_grid[new_r][new_c] = max(new_grid[new_r][new_c], color)

    # Step 8: Create and return the result
    return ColoredGrid(values=new_grid)
