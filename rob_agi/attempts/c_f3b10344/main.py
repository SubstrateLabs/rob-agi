from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_f3b10344(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f3b10344 challenge by creating a sky blue network that connects and surrounds shapes of the same color.
    
    The function identifies non-black shapes in the grid, creates a mask for each color that includes
    and surrounds the shapes, fills the mask with sky blue (8), and preserves the original non-black cells.
    The result is a grid where shapes of the same color are connected by sky blue, and all non-black
    areas are surrounded by sky blue.
    """
    if all(cell == 0 for row in input_grid.values for cell in row):
        return input_grid

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    colors = set(cell for row in grid.values for cell in row if cell != 0)

    def flood_fill(r: int, c: int, target_color: int, replacement_color: int) -> Set[Tuple[int, int]]:
        if target_color == replacement_color:
            return set()
        shape = set()
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == target_color:
                grid.values[r][c] = replacement_color
                shape.add((r, c))
                stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        return shape

    def create_mask(color: int) -> List[List[bool]]:
        mask = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == color:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                mask[nr][nc] = True
        return mask

    def connect_mask(mask: List[List[bool]]):
        def flood_fill_mask(r: int, c: int):
            stack = [(r, c)]
            while stack:
                r, c = stack.pop()
                if 0 <= r < rows and 0 <= c < cols and not mask[r][c]:
                    mask[r][c] = True
                    stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

        for r in range(rows):
            for c in range(cols):
                if mask[r][c]:
                    flood_fill_mask(r, c)
                    return

    for color in colors:
        mask = create_mask(color)
        connect_mask(mask)
        for r in range(rows):
            for c in range(cols):
                if mask[r][c] and grid.values[r][c] == 0:
                    grid.values[r][c] = 8

    # Expand sky blue regions
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 0:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 8:
                        grid.values[r][c] = 8
                        break

    # Preserve original shapes
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] != 0:
                grid.values[r][c] = input_grid.values[r][c]

    return grid
