from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ea959feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying horizontal and vertical color sequences
    and applying them to create a consistent pattern across the entire grid.
    
    The solution works as follows:
    1. Identifies the horizontal color sequence from the rows of the input grid.
    2. Identifies the vertical color sequence from the columns of the input grid.
    3. Creates a function to determine the correct color for any cell based on these sequences.
    4. Generates a new grid by applying this function to each cell.
    5. Returns a new ColoredGrid with the corrected pattern.
    
    This approach works for all cases by identifying the underlying pattern in each input
    and extending it consistently across the entire grid.
    """
    horizontal_sequence = identify_sequence(input_grid.values)
    vertical_sequence = identify_sequence(list(zip(*input_grid.values)))
    corrected_values = correct_grid(input_grid, horizontal_sequence, vertical_sequence)
    return ColoredGrid(values=corrected_values)

def identify_sequence(lines: List[List[int]]) -> List[int]:
    """Identifies the repeating color sequence in a list of lines."""
    for line in lines:
        for i in range(1, len(line) // 2 + 1):
            if line[:i] * (len(line) // i) == line[:len(line) - (len(line) % i)]:
                return line[:i]
    return lines[0]  # Fallback to the first line if no clear repetition is found

def get_correct_color(row: int, col: int, horizontal_seq: List[int], vertical_seq: List[int]) -> int:
    """Determines the correct color for a given cell based on the horizontal and vertical sequences."""
    horizontal_color = horizontal_seq[col % len(horizontal_seq)]
    vertical_color = vertical_seq[row % len(vertical_seq)]
    return (horizontal_color + vertical_color) % 10

def correct_grid(input_grid: ColoredGrid, horizontal_seq: List[int], vertical_seq: List[int]) -> List[List[int]]:
    """Generates a new grid by applying the identified pattern."""
    rows, cols = input_grid.get_dimensions()
    return [[get_correct_color(i, j, horizontal_seq, vertical_seq) for j in range(cols)] for i in range(rows)]
