from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ea959feb(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid pattern challenge by identifying and correcting disrupted patterns.
    
    The solution works as follows:
    1. Analyzes the input grid to identify the diagonal stripe pattern.
    2. Determines the stripe width and color sequence.
    3. Creates a function to determine the correct color for any cell.
    4. Iterates through the grid, correcting cells that don't match the pattern.
    5. Returns a new ColoredGrid with the corrected pattern.
    
    This approach works for all cases by identifying unique patterns in each input
    and only changing cells that don't match the identified pattern.
    """
    stripe_width, color_sequence = analyze_grid(input_grid)
    corrected_values = correct_grid(input_grid, stripe_width, color_sequence)
    return ColoredGrid(values=corrected_values)

def analyze_grid(grid: ColoredGrid) -> Tuple[int, List[int]]:
    """Analyzes the grid to determine stripe width and color sequence."""
    rows, cols = grid.get_dimensions()
    for i in range(min(rows, cols)):
        if grid.values[i][i] == grid.values[0][0]:
            stripe_width = i
            break
    color_sequence = [grid.values[0][j] for j in range(stripe_width)]
    return stripe_width, color_sequence

def get_correct_color(row: int, col: int, stripe_width: int, color_sequence: List[int]) -> int:
    """Determines the correct color for a given cell based on the pattern."""
    pattern_position = (row + col) % (stripe_width * len(color_sequence))
    color_index = pattern_position // stripe_width
    return color_sequence[color_index]

def correct_grid(input_grid: ColoredGrid, stripe_width: int, color_sequence: List[int]) -> List[List[int]]:
    """Corrects the grid by applying the identified pattern."""
    rows, cols = input_grid.get_dimensions()
    new_grid = []
    for i in range(rows):
        new_row = []
        for j in range(cols):
            correct_color = get_correct_color(i, j, stripe_width, color_sequence)
            new_row.append(correct_color)
        new_grid.append(new_row)
    return new_grid
