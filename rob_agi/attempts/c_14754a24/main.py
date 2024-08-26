from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import heapq

def solve_14754a24(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by creating optimal L-shaped patterns around yellow squares.
    
    1. Creates a deep copy of the input grid to modify.
    2. Identifies all yellow (4) squares in the grid.
    3. Generates and scores all possible L-shapes for each yellow square.
    4. Applies L-shapes using a priority queue based on scores.
    5. Connects nearby L-shapes and fills gaps to create larger patterns.
    6. Handles isolated yellow squares with partial L-shapes.
    7. Optimizes coverage by extending existing red regions.
    8. Performs multiple optimization passes to improve the solution.
    9. Validates the final grid to ensure all red squares are part of valid L-shapes.
    
    Returns a new ColoredGrid with the transformed values.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def find_yellow_squares() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 4]
    
    def get_possible_l_shapes(r: int, c: int) -> List[List[Tuple[int, int]]]:
        shapes = []
        for length1 in range(2, 7):  # First arm of L-shape
            for length2 in range(2, 7):  # Second arm of L-shape
                for dr1, dc1 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
                            shape = [(r + dr1*i, c + dc1*i) for i in range(length1)] + \
                                    [(r + dr2*i, c + dc2*i) for i in range(1, length2)]
                            if all(0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in [0, 4, 5] for nr, nc in shape):
                                shapes.append(shape)
        return shapes
    
    def score_l_shape(shape: List[Tuple[int, int]]) -> int:
        score = len(shape)  # Base score is the size of the shape
        yellow_count = sum(1 for r, c in shape if grid.values[r][c] == 4)
        adjacent_red = sum(1 for r, c in shape for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2)
        connected_yellows = len(set((r, c) for r, c in shape if grid.values[r][c] == 4))
        return score + yellow_count * 3 + adjacent_red + connected_yellows * 2
    
    def apply_l_shape(shape: List[Tuple[int, int]]) -> None:
        for r, c in shape:
            if grid.values[r][c] != 4:  # Don't convert yellow squares
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
                if not any(grid.values[r][c] == 2 for r, c in shape if grid.values[r][c] != 4):
                    apply_l_shape(shape)
                    covered_yellows.update(yellows_in_shape)
    
    def optimize_pattern() -> None:
        for _ in range(3):  # Multiple optimization passes
            for r in range(rows):
                for c in range(cols):
                    if grid.values[r][c] == 2:
                        neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                        if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] in [2, 4])
                        if neighbors == 0:
                            grid.values[r][c] = 0  # Remove isolated red squares
            
            # Try to extend L-shapes
            for r in range(rows):
                for c in range(cols):
                    if grid.values[r][c] == 4:
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 0:
                                grid.values[r+dr][c+dc] = 2  # Extend L-shape
    
    def verify_and_extend_l_shapes() -> bool:
        valid = True
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    if not any(grid.values[r+dr][c+dc] == 4 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        valid = False
                        grid.values[r][c] = 0  # Remove invalid red square
                elif grid.values[r][c] in [0, 5]:
                    # Try to extend L-shapes
                    red_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                        if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2)
                    yellow_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                           if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 4)
                    if red_neighbors >= 1 and yellow_neighbors >= 1:
                        grid.values[r][c] = 2  # Extend L-shape
        return valid

    def connect_nearby_l_shapes() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] in [0, 5]:
                    red_neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                     if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2]
                    if len(red_neighbors) >= 2:
                        yellow_nearby = any(grid.values[nr+dr][nc+dc] == 4
                                            for nr, nc in red_neighbors
                                            for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                                            if 0 <= nr+dr < rows and 0 <= nc+dc < cols)
                        if yellow_nearby:
                            grid.values[r][c] = 2  # Connect nearby L-shapes
    
    def handle_isolated_yellows() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 4:
                    red_neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                        if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2)
                    if red_neighbors == 0:
                        # Create a partial L-shape
                        for dr1, dc1 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
                                    if (0 <= r+dr1 < rows and 0 <= c+dc1 < cols and grid.values[r+dr1][c+dc1] in [0, 5] and
                                        0 <= r+dr2 < rows and 0 <= c+dc2 < cols and grid.values[r+dr2][c+dc2] in [0, 5]):
                                        grid.values[r+dr1][c+dc1] = 2
                                        grid.values[r+dr2][c+dc2] = 2
                                        return  # Only create one partial L-shape per isolated yellow
    
    process_yellow_squares()
    optimize_pattern()
    
    for _ in range(5):  # Multiple optimization passes
        verify_and_extend_l_shapes()
        connect_nearby_l_shapes()
        handle_isolated_yellows()
        optimize_pattern()
    
    if verify_and_extend_l_shapes():
        return grid
    else:
        return input_grid  # Return original grid if verification fails
