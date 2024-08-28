from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import heapq

def solve_14754a24(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by creating optimal L-shaped patterns around yellow squares.
    
    1. Analyzes the input grid to identify yellow squares and their distribution.
    2. Generates possible L-shapes of varying sizes for each yellow square.
    3. Scores L-shapes based on coverage, efficiency, and adaptability.
    4. Places L-shapes prioritizing higher scores and yellow square coverage.
    5. Optimizes for clusters by considering interlocking L-shapes.
    6. Handles linear patterns of yellow squares with connected smaller L-shapes.
    7. Covers isolated yellow squares by extending nearby L-shapes or creating minimal new ones.
    8. Performs global optimization to improve overall coverage and efficiency.
    9. Cleans up the solution by removing invalid red squares and ensuring L-shape validity.
    10. Makes a final pass to maximize yellow square coverage.
    
    Returns a new ColoredGrid with the transformed values.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def find_yellow_squares() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 4]
    
    def get_possible_l_shapes(r: int, c: int) -> List[List[Tuple[int, int]]]:
        shapes = []
        for length1 in range(2, 8):  # Increased max length
            for length2 in range(2, 8):  # Increased max length
                for dr1, dc1 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
                            shape = [(r + dr1*i, c + dc1*i) for i in range(length1)] + \
                                    [(r + dr2*i, c + dc2*i) for i in range(1, length2)]
                            if all(0 <= nr < rows and 0 <= nc < cols for nr, nc in shape):
                                if all(grid.values[nr][nc] in [0, 4, 5] or (nr, nc) == (r, c) for nr, nc in shape):
                                    shapes.append(shape)
        return shapes
    
    def score_l_shape(shape: List[Tuple[int, int]]) -> float:
        yellow_count = sum(1 for r, c in shape if grid.values[r][c] == 4)
        efficiency = yellow_count / len(shape)
        adaptability = sum(1 for r, c in shape for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 4)
        return yellow_count * 10 + efficiency * 5 + adaptability * 2
    
    def apply_l_shape(shape: List[Tuple[int, int]]) -> None:
        for r, c in shape:
            if grid.values[r][c] in [0, 5]:  # Only convert black or gray squares
                grid.values[r][c] = 2
    
    def process_yellow_squares() -> None:
        yellow_squares = find_yellow_squares()
        all_shapes = []
        for r, c in yellow_squares:
            possible_shapes = get_possible_l_shapes(r, c)
            for shape in possible_shapes:
                score = score_l_shape(shape)
                heapq.heappush(all_shapes, (-score, shape))  # Use negative score for max-heap
        
        covered_yellows = set()
        while all_shapes:
            _, shape = heapq.heappop(all_shapes)
            yellows_in_shape = set((r, c) for r, c in shape if grid.values[r][c] == 4)
            if yellows_in_shape - covered_yellows:
                if not any(grid.values[r][c] == 2 for r, c in shape if grid.values[r][c] not in [2, 4]):
                    apply_l_shape(shape)
                    covered_yellows.update(yellows_in_shape)
    
    def optimize_clusters() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 4:
                    cluster = [(r, c)]
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 4 and (nr, nc) not in cluster:
                                cluster.append((nr, nc))
                                stack.append((nr, nc))
                    if len(cluster) > 1:
                        optimize_cluster(cluster)
    
    def optimize_cluster(cluster: List[Tuple[int, int]]) -> None:
        best_shapes = []
        for r, c in cluster:
            shapes = get_possible_l_shapes(r, c)
            best_shape = max(shapes, key=score_l_shape)
            best_shapes.append(best_shape)
        
        covered = set()
        for shape in best_shapes:
            yellows = set((r, c) for r, c in shape if grid.values[r][c] == 4)
            if yellows - covered:
                apply_l_shape(shape)
                covered.update(yellows)
    
    def handle_linear_patterns() -> None:
        for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for r in range(rows):
                for c in range(cols):
                    if grid.values[r][c] == 4:
                        line = [(r, c)]
                        nr, nc = r + direction[0], c + direction[1]
                        while 0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] == 4:
                            line.append((nr, nc))
                            nr, nc = nr + direction[0], nc + direction[1]
                        if len(line) > 2:
                            handle_line(line)
    
    def handle_line(line: List[Tuple[int, int]]) -> None:
        for i in range(0, len(line) - 1, 2):
            r1, c1 = line[i]
            r2, c2 = line[i + 1]
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                if 0 <= r1 + dr < rows and 0 <= c1 + dc < cols and grid.values[r1 + dr][c1 + dc] in [0, 5]:
                    grid.values[r1 + dr][c1 + dc] = 2
                    break
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                if 0 <= r2 + dr < rows and 0 <= c2 + dc < cols and grid.values[r2 + dr][c2 + dc] in [0, 5]:
                    grid.values[r2 + dr][c2 + dc] = 2
                    break
    
    def handle_isolated_yellows() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 4:
                    if not any(grid.values[r+dr][c+dc] == 2 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        for dr1, dc1 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
                                    if (0 <= r+dr1 < rows and 0 <= c+dc1 < cols and grid.values[r+dr1][c+dc1] in [0, 5] and
                                        0 <= r+dr2 < rows and 0 <= c+dc2 < cols and grid.values[r+dr2][c+dc2] in [0, 5]):
                                        grid.values[r+dr1][c+dc1] = 2
                                        grid.values[r+dr2][c+dc2] = 2
                                        break
    
    def global_optimization() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    if not any(grid.values[r+dr][c+dc] == 4 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        grid.values[r][c] = 0  # Remove red square if it's not part of a valid L-shape
    
    def cleanup_and_validate() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    if not any(grid.values[r+dr][c+dc] == 4 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        grid.values[r][c] = 0  # Remove invalid red square
    
    process_yellow_squares()
    optimize_clusters()
    handle_linear_patterns()
    handle_isolated_yellows()
    global_optimization()
    cleanup_and_validate()
    
    return grid
