from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_93c31fbe(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a balanced and symmetric pattern of blue (1) pixels
    while respecting other colored elements. The solution:
    1. Identifies all non-black structures in the grid.
    2. Creates connections between structures of the same color, prioritizing diagonal paths.
    3. Balances the overall pattern by mirroring blue pixel placements.
    4. Completes partial blue structures without interfering with existing non-black pixels.
    5. Ensures connectivity and symmetry in the final blue pixel pattern.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_structures() -> Dict[int, List[Set[Tuple[int, int]]]]:
        structures = {color: [] for color in range(1, 10)}  # Exclude black (0)
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and grid.values[r][c] != 0:
                    color = grid.values[r][c]
                    structure = set()
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) not in visited and grid.values[cr][cc] == color:
                            visited.add((cr, cc))
                            structure.add((cr, cc))
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                                nr, nc = cr + dr, cc + dc
                                if is_valid(nr, nc):
                                    stack.append((nr, nc))
                    structures[color].append(structure)
        return structures

    def find_connection_points(structures: Dict[int, List[Set[Tuple[int, int]]]]) -> List[Tuple[int, int, int, int]]:
        connections = []
        for color, color_structures in structures.items():
            for i, struct1 in enumerate(color_structures):
                for j, struct2 in enumerate(color_structures[i+1:], start=i+1):
                    min_dist = float('inf')
                    best_connection = None
                    for r1, c1 in struct1:
                        for r2, c2 in struct2:
                            dist = max(abs(r2 - r1), abs(c2 - c1))
                            if dist < min_dist:
                                min_dist = dist
                                best_connection = (r1, c1, r2, c2)
                    if best_connection:
                        connections.append(best_connection)
        return sorted(connections, key=lambda x: max(abs(x[2] - x[0]), abs(x[3] - x[1])))

    def connect_points(r1: int, c1: int, r2: int, c2: int):
        dr = 1 if r2 > r1 else -1 if r2 < r1 else 0
        dc = 1 if c2 > c1 else -1 if c2 < c1 else 0
        r, c = r1, c1
        while (r, c) != (r2, c2):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 1
            if r != r2:
                r += dr
            if c != c2:
                c += dc

    def mirror_placement(r: int, c: int):
        mirror_r, mirror_c = rows - 1 - r, cols - 1 - c
        if is_valid(mirror_r, mirror_c) and grid.values[mirror_r][mirror_c] == 0:
            grid.values[mirror_r][mirror_c] = 1

    def complete_structures():
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 1:
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nr, nc = r + dr, c + dc
                        if is_valid(nr, nc) and grid.values[nr][nc] == 0:
                            blue_neighbors = sum(1 for d in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid(nr+d[0], nc+d[1]) and grid.values[nr+d[0]][nc+d[1]] == 1)
                            if blue_neighbors >= 2:
                                grid.values[nr][nc] = 1
                                mirror_placement(nr, nc)

    # Main execution
    structures = get_structures()
    connections = find_connection_points(structures)

    for r1, c1, r2, c2 in connections:
        connect_points(r1, c1, r2, c2)

    complete_structures()

    return grid
