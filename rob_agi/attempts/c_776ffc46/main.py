
from rob_agi.colored_grid import ColoredGrid
import copy

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    output_grid = copy.deepcopy(input_grid)
    rows, cols = len(input_grid.values), len(input_grid.values[0])

    def is_part_of_larger_shape(r, c, visited):
        if (r, c) in visited:
            return False
        visited.add((r, c))
        
        count = 1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and input_grid.values[nr][nc] == 1:
                count += is_part_of_larger_shape(nr, nc, visited)
        
        return count

    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:
                visited = set()
                if is_part_of_larger_shape(r, c, visited) > 1:
                    for vr, vc in visited:
                        output_grid.values[vr][vc] = 2

    return output_grid
