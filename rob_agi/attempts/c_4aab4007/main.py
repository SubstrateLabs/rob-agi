from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Generator

def detect_pattern_sequence(grid: ColoredGrid) -> List[int]:
    """Detect the pattern sequence in the input grid."""
    pattern = []
    for row in range(2, grid.num_rows):
        for col in range(2, grid.num_cols):
            if grid[row][col] not in [0, 1, 4] and grid[row][col] not in pattern:
                pattern.append(grid[row][col])
    return pattern

def spiral_coordinates(rows: int, cols: int) -> Generator[Tuple[int, int], None, None]:
    """Generate coordinates in a spiral pattern."""
    top, bottom, left, right = 2, rows-1, 2, cols-1
    while top <= bottom and left <= right:
        for i in range(left, right + 1):
            yield top, i
        top += 1
        for i in range(top, bottom + 1):
            yield i, right
        right -= 1
        if top <= bottom:
            for i in range(right, left - 1, -1):
                yield bottom, i
            bottom -= 1
        if left <= right:
            for i in range(bottom, top - 1, -1):
                yield i, left
            left += 1

def solve_4aab4007(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by filling black areas with a detected pattern.
    
    The function identifies the pattern sequence from non-black, non-blue, and non-yellow cells
    in the input grid. It then creates a new grid with the same blue border and yellow frame,
    and fills the inner area using a spiral pattern. Black cells are replaced with the next
    number in the pattern sequence, while other cells maintain their original values.
    The pattern is synchronized with existing non-black cells to maintain consistency.
    """
    pattern_sequence = detect_pattern_sequence(input_grid)
    output_grid = input_grid.deep_copy()
    sequence_pointer = 0
    
    for row, col in spiral_coordinates(input_grid.num_rows, input_grid.num_cols):
        if input_grid[row][col] == 0:
            output_grid.values[row][col] = pattern_sequence[sequence_pointer]
            sequence_pointer = (sequence_pointer + 1) % len(pattern_sequence)
        elif input_grid[row][col] not in [1, 4]:
            output_grid.values[row][col] = input_grid[row][col]
            sequence_pointer = (pattern_sequence.index(input_grid[row][col]) + 1) % len(pattern_sequence)
    
    return output_grid
