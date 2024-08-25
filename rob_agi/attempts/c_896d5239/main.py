from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

from typing import List, Tuple, Set
from collections import deque

def solve_896d5239(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by finding non-overlapping rectangular regions around green (3) squares
    and filling them with sky blue (8), while preserving the original green squares and blue (1) squares.
    
    The algorithm works as follows:
    1. Identify all green squares in the grid.
    2. Find connected regions of black (0) and green (3) squares.
    3. Create potential rectangles for each region, prioritizing those with more green squares.
    4. Sort rectangles by the number of green squares they contain and their area.
    5. Apply sky blue rectangles without overlapping, adjusting if necessary.
    6. Restore the original green squares.

    This approach ensures that regions with multiple green squares are prioritized,
    maximizes the coverage of black areas between green squares,
    and preserves the original pattern of blue squares.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    green_squares = set((r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 3)
    processed = set()

    def find_connected_region(start_r: int, start_c: int) -> Set[Tuple[int, int]]:
        region = set()
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) not in region and 0 <= r < rows and 0 <= c < cols and output_grid.get_cell(r, c) in [0, 3]:
                region.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    queue.append((r + dr, c + dc))
        return region

    def create_rectangle(region: Set[Tuple[int, int]]) -> Tuple[int, int, int, int, int]:
        green_in_region = region & green_squares
        min_r = min(r for r, _ in green_in_region)
        max_r = max(r for r, _ in green_in_region)
        min_c = min(c for _, c in green_in_region)
        max_c = max(c for _, c in green_in_region)
        
        # Expand rectangle
        while min_r > 0 and all(output_grid.get_cell(min_r-1, c) in [0, 3] for c in range(min_c, max_c+1)):
            min_r -= 1
        while max_r < rows-1 and all(output_grid.get_cell(max_r+1, c) in [0, 3] for c in range(min_c, max_c+1)):
            max_r += 1
        while min_c > 0 and all(output_grid.get_cell(r, min_c-1) in [0, 3] for r in range(min_r, max_r+1)):
            min_c -= 1
        while max_c < cols-1 and all(output_grid.get_cell(r, max_c+1) in [0, 3] for r in range(min_r, max_r+1)):
            max_c += 1
        
        return (min_r, min_c, max_r, max_c, len(green_in_region))

    rectangles = []
    for r, c in green_squares:
        if (r, c) not in processed:
            region = find_connected_region(r, c)
            rectangles.append(create_rectangle(region))
            processed.update(region)

    rectangles.sort(key=lambda x: (x[4], (x[2]-x[0]+1)*(x[3]-x[1]+1)), reverse=True)

    filled = set()
    for min_r, min_c, max_r, max_c, _ in rectangles:
        if not any((r, c) in filled for r in range(min_r, max_r+1) for c in range(min_c, max_c+1)):
            for r in range(min_r, max_r+1):
                for c in range(min_c, max_c+1):
                    if output_grid.get_cell(r, c) != 3:
                        output_grid.set_cell(r, c, 8)
                    filled.add((r, c))

    for r, c in green_squares:
        output_grid.set_cell(r, c, 3)

    return output_grid
