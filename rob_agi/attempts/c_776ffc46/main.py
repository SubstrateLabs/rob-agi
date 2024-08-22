from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions that form a single straight line (horizontal, vertical, or diagonal) remain blue.
    2. Blue (1) regions that do not form a single straight line (including crosses, T-shapes, and L-shapes) change to red (2).
    3. All other colors remain unchanged.
    4. Gray (5) regions act as boundaries or "boxes" that isolate different areas of the grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_straight_line(region: Set[Tuple[int, int]]) -> bool:
        if len(region) <= 2:
            return True
    
        points = sorted(list(region))
        first, last = points[0], points[-1]
        dx, dy = last[0] - first[0], last[1] - first[1]
    
        if dx == 0:  # Vertical line
            return all(p[0] == first[0] for p in points)
        elif dy == 0:  # Horizontal line
            return all(p[1] == first[1] for p in points)
        elif abs(dx) == abs(dy):  # Diagonal line
            return all((p[0] - first[0]) * dy == (p[1] - first[1]) * dx for p in points)
    
        return False

    def flood_fill(r: int, c: int, compartment: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        region = set()
        stack = [(r, c)]
        color = output_grid.get_cell(r, c)
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) in compartment and output_grid.get_cell(curr_r, curr_c) == color:
                region.add((curr_r, curr_c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Only orthogonal neighbors
                    new_r, new_c = curr_r + dr, curr_c + dc
                    if (new_r, new_c) in compartment:
                        stack.append((new_r, new_c))
        return region

    def find_compartments() -> List[Set[Tuple[int, int]]]:
        compartments = []
        visited = set()

        def dfs(r: int, c: int) -> Set[Tuple[int, int]]:
            compartment = set()
            stack = [(r, c)]
            while stack:
                curr_r, curr_c = stack.pop()
                if (curr_r, curr_c) not in visited and output_grid.get_cell(curr_r, curr_c) != 5:
                    visited.add((curr_r, curr_c))
                    compartment.add((curr_r, curr_c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        new_r, new_c = curr_r + dr, curr_c + dc
                        if 0 <= new_r < rows and 0 <= new_c < cols:
                            stack.append((new_r, new_c))
            return compartment

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and output_grid.get_cell(r, c) != 5:
                    compartment = dfs(r, c)
                    if compartment:
                        compartments.append(compartment)

        return compartments

    compartments = find_compartments()

    for compartment in compartments:
        compartment_visited = set()
        for r, c in compartment:
            if (r, c) not in compartment_visited and output_grid.get_cell(r, c) == 1:  # Blue
                region = flood_fill(r, c, compartment)
                compartment_visited.update(region)
                if len(region) > 1 and not is_straight_line(region):
                    for cell_r, cell_c in region:
                        output_grid.set_cell(cell_r, cell_c, 2)  # Change to Red (2)
                # Single blue cells and straight lines remain unchanged

    return output_grid
