from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_52fd389e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by processing yellow (4), sky (8), and magenta (6) regions.
    1. Yellow regions are surrounded by a border of their internal color or blue.
    2. Sky regions are expanded by 2 cells in all directions, not overwriting yellow cells.
    3. Magenta regions are expanded until reaching non-black cells or grid edges.
    4. Remaining black cells surrounded by non-black cells are filled with the majority color.
    5. A final pass ensures sky cells don't overwrite yellow borders inappropriately.
    Steps are applied in the order listed above.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid(r+dr, c+dc)]

    def flood_fill(r: int, c: int, color: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        region = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == color:
                region.append((curr_r, curr_c))
                visited.add((curr_r, curr_c))
                stack.extend(get_neighbors(curr_r, curr_c))
        return region

    def process_yellow_regions():
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) == 4:
                    region = flood_fill(r, c, 4, visited)
                    border_color = 1
                    for yr, yc in region:
                        for nr, nc in get_neighbors(yr, yc):
                            if grid.get_cell(nr, nc) not in [4, 0]:
                                border_color = grid.get_cell(nr, nc)
                                break
                        if border_color != 1:
                            break
                    for yr, yc in region:
                        for nr, nc in get_neighbors(yr, yc):
                            if grid.get_cell(nr, nc) == 0:
                                grid.set_cell(nr, nc, border_color)

    def process_sky_regions():
        sky_expansion = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8:
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if is_valid(r+dr, c+dc) and grid.get_cell(r+dr, c+dc) not in [4, 8]:
                                sky_expansion.set_cell(r+dr, c+dc, 8)
        
        for r in range(rows):
            for c in range(cols):
                if sky_expansion.get_cell(r, c) == 8 and grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 8)

    def process_magenta_regions():
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.get_cell(r, c) == 6:
                    region = flood_fill(r, c, 6, visited)
                    expanded_region = set(region)
                    while True:
                        new_cells = set()
                        for mr, mc in expanded_region:
                            for nr, nc in get_neighbors(mr, mc):
                                if grid.get_cell(nr, nc) == 0:
                                    new_cells.add((nr, nc))
                        if not new_cells:
                            break
                        expanded_region.update(new_cells)
                    for mr, mc in expanded_region:
                        grid.set_cell(mr, mc, 6)

    def clean_up_black_cells():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0:
                    neighbors = get_neighbors(r, c)
                    colors = [grid.get_cell(nr, nc) for nr, nc in neighbors if grid.get_cell(nr, nc) != 0]
                    if colors:
                        grid.set_cell(r, c, max(set(colors), key=colors.count))

    def final_pass():
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 8:
                    neighbors = get_neighbors(r, c)
                    for nr, nc in neighbors:
                        if grid.get_cell(nr, nc) == 4:
                            yellow_neighbors = [
                                (nnr, nnc) for nnr, nnc in get_neighbors(nr, nc)
                                if grid.get_cell(nnr, nnc) == 4
                            ]
                            if len(yellow_neighbors) == 0:
                                grid.set_cell(r, c, grid.get_cell(nr, nc))
                            break

    process_yellow_regions()
    process_sky_regions()
    process_magenta_regions()
    clean_up_black_cells()
    final_pass()

    return grid
