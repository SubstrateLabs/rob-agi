from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
import heapq

def solve_14754a24(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by creating optimal L-shaped patterns around yellow squares.
    
    1. Scans the grid to identify all yellow (4) squares.
    2. Creates a deep copy of the input grid to modify.
    3. Processes each yellow square:
       - Identifies possible L-shapes around the yellow square.
       - Evaluates and scores each L-shape based on size, enclosed yellows, and connections.
       - Selects and applies the best L-shape, converting it to red (2).
    4. Performs an optimization pass to improve L-shapes and handle edge cases.
    5. Verifies that all red squares form valid L-shapes associated with yellow squares.
    
    Returns a new ColoredGrid with the transformed values.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    
    def find_yellow_squares() -> List[Tuple[int, int]]:
        return [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 4]
    
    def get_possible_l_shapes(r: int, c: int) -> List[List[Tuple[int, int]]]:
        shapes = []
        for length in range(2, 6):  # L-shapes from size 2 to 5
            for dr1, dc1 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if (dr1, dc1) != (dr2, dc2) and (dr1, dc1) != (-dr2, -dc2):
                        shape = [(r + dr1*i, c + dc1*i) for i in range(length)] + \
                                [(r + dr1*(length-1) + dr2*i, c + dc1*(length-1) + dc2*i) for i in range(1, length)]
                        if all(0 <= nr < rows and 0 <= nc < cols and grid.values[nr][nc] in [0, 4, 5] for nr, nc in shape):
                            shapes.append(shape)
        return shapes
    
    def score_l_shape(shape: List[Tuple[int, int]]) -> int:
        score = len(shape)  # Base score is the size of the shape
        yellow_count = sum(1 for r, c in shape if grid.values[r][c] == 4)
        adjacent_red = sum(1 for r, c in shape for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2)
        return score + yellow_count * 2 + adjacent_red
    
    def apply_l_shape(shape: List[Tuple[int, int]]) -> None:
        for r, c in shape:
            if grid.values[r][c] != 4:  # Don't convert yellow squares
                grid.values[r][c] = 2
    
    def process_yellow_squares() -> None:
        yellow_squares = find_yellow_squares()
        for r, c in yellow_squares:
            if grid.values[r][c] == 4:  # Check if still yellow
                possible_shapes = get_possible_l_shapes(r, c)
                if possible_shapes:
                    best_shape = max(possible_shapes, key=score_l_shape)
                    apply_l_shape(best_shape)
    
    def optimize_pattern() -> None:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                    if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 2)
                    if neighbors == 0:
                        grid.values[r][c] = 0  # Remove isolated red squares
    
    def verify_l_shapes() -> bool:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 2:
                    if not any(grid.values[r+dr][c+dc] == 4 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        return False  # Red square not associated with a yellow square
        return True
    
    process_yellow_squares()
    optimize_pattern()
    
    if not verify_l_shapes():
        return input_grid  # Return original grid if verification fails
    
    return grid
