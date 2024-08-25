from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Create a distance map for each black cell to the nearest non-black cell.
    2. Identify pattern sources around black regions.
    3. Generate extended patterns from these sources.
    4. Fill black regions using the extended patterns, considering distance and direction.
    5. Resolve conflicts between patterns and create smooth transitions.
    6. Preserve and integrate any existing colored structures within black regions.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def create_distance_map() -> Dict[Tuple[int, int], int]:
        distance_map = {}
        queue = deque()
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) != 0:
                    distance_map[(r, c)] = 0
                    queue.append((r, c, 0))
        
        while queue:
            r, c, dist = queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in distance_map:
                    distance_map[(nr, nc)] = dist + 1
                    queue.append((nr, nc, dist + 1))
        
        return distance_map

    def get_pattern(r: int, c: int, direction: Tuple[int, int], length: int) -> List[int]:
        pattern = []
        for _ in range(length):
            if 0 <= r < rows and 0 <= c < cols and grid.get_cell(r, c) != 0:
                pattern.append(grid.get_cell(r, c))
            r += direction[0]
            c += direction[1]
        return pattern

    def fill_cell(r: int, c: int, distance_map: Dict[Tuple[int, int], int]):
        if grid.get_cell(r, c) != 0:
            return

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        patterns = []
        for dr, dc in directions:
            pattern = get_pattern(r + dr, c + dc, (dr, dc), 5)
            if pattern:
                patterns.append((pattern, distance_map.get((r + dr, c + dc), float('inf'))))
        
        if patterns:
            patterns.sort(key=lambda x: x[1])
            chosen_pattern = patterns[0][0]
            grid.set_cell(r, c, chosen_pattern[0])
        else:
            grid.set_cell(r, c, 1)  # Default to color 1 if no pattern found

    distance_map = create_distance_map()
    
    for _ in range(2):  # Repeat twice to ensure all cells are filled
        for r in range(rows):
            for c in range(cols):
                fill_cell(r, c, distance_map)

    return grid
