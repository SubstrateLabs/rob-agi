from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5af49b42(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding colored dots based on a sequence.
    
    1. Extracts the expansion sequence from the bottom-right corner.
    2. Identifies expandable dots (non-zero cells surrounded by zeros).
    3. For each dot, determines expansion direction and length (3 or 4).
    4. Applies expansions to create a new grid.
    5. Returns the transformed grid.
    """
    def get_expansion_sequence(grid: ColoredGrid) -> List[int]:
        last_row = grid.values[-1]
        return [color for color in last_row if color != 0]

    def find_expandable_dots(grid: ColoredGrid) -> List[Tuple[int, int]]:
        dots = []
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 0:
                    if all(grid.values[r+dr][c+dc] == 0 
                           for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        dots.append((r, c))
        return dots

    def determine_expansion_details(grid: ColoredGrid, dot: Tuple[int, int], sequence: List[int]) -> Tuple[int, int, int]:
        r, c = dot
        rows, cols = grid.get_dimensions()
        right_space = next((i for i in range(c+1, cols) if grid.values[r][i] != 0), cols) - c - 1
        left_space = c - next((i for i in range(c-1, -1, -1) if grid.values[r][i] != 0), -1) - 1
        
        length = min(max(3, 4), max(right_space, left_space))
        direction = 1 if right_space >= left_space else -1
        return length, direction, grid.values[r][c]

    def create_expansion(sequence: List[int], length: int, start_color: int) -> List[int]:
        start_index = sequence.index(start_color)
        expansion = sequence[start_index:start_index+length]
        return expansion + [0] * (length - len(expansion))

    def apply_expansion(grid: ColoredGrid, dot: Tuple[int, int], expansion: List[int], direction: int) -> None:
        r, c = dot
        for i, color in enumerate(expansion):
            grid.values[r][c + i*direction] = color

    sequence = get_expansion_sequence(input_grid)
    dots = find_expandable_dots(input_grid)
    new_grid = input_grid.deep_copy()

    for dot in dots:
        length, direction, start_color = determine_expansion_details(new_grid, dot, sequence)
        expansion = create_expansion(sequence, length, start_color)
        apply_expansion(new_grid, dot, expansion, direction)

    return new_grid
