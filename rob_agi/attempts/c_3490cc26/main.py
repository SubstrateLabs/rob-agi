from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_3490cc26(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting sky blue (8) squares and red (2) squares with orange (7) paths.
    
    The solution follows these steps:
    1. Find all 2x2 sky blue squares and the 2x2 red square in the input grid.
    2. Generate all pairs of squares (blue-blue and blue-red).
    3. For each pair, fill the rectangular area between them with orange (7), except for the blue and red squares.
    4. Handle special cases like single blue square with red square.
    5. Validate the final grid to ensure all areas between squares are filled correctly.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with orange paths connecting blue and red squares.
    """
    squares = find_squares(input_grid)
    output_grid = input_grid.deep_copy()
    pairs = generate_pairs(squares)
    for start, end in pairs:
        fill_rectangle(output_grid, start, end)
    if len(squares['blue']) == 1 and squares['red']:
        fill_rectangle(output_grid, squares['blue'][0], squares['red'])
    validate_grid(output_grid, squares)
    return output_grid

def find_squares(grid: ColoredGrid) -> Dict[str, List[Tuple[int, int]]]:
    """Find all 2x2 sky blue squares and the 2x2 red square in the grid."""
    squares = {'blue': [], 'red': []}
    rows, cols = grid.get_dimensions()
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(grid.get_cell(r+dr, c+dc) == 8 for dr in range(2) for dc in range(2)):
                squares['blue'].append((r, c))
            elif all(grid.get_cell(r+dr, c+dc) == 2 for dr in range(2) for dc in range(2)):
                squares['red'].append((r, c))
    return squares

def generate_pairs(squares: Dict[str, List[Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Generate all pairs of squares (blue-blue and blue-red)."""
    pairs = []
    for i in range(len(squares['blue'])):
        for j in range(i+1, len(squares['blue'])):
            pairs.append((squares['blue'][i], squares['blue'][j]))
        for red_square in squares['red']:
            pairs.append((squares['blue'][i], red_square))
    return pairs

def fill_rectangle(grid: ColoredGrid, start: Tuple[int, int], end: Tuple[int, int]):
    """Fill the rectangular area between start and end with orange (7), except for blue and red squares."""
    for r in range(min(start[0], end[0]), max(start[0], end[0]) + 2):
        for c in range(min(start[1], end[1]), max(start[1], end[1]) + 2):
            if grid.get_cell(r, c) not in [2, 8]:
                grid.set_cell(r, c, 7)

def validate_grid(grid: ColoredGrid, squares: Dict[str, List[Tuple[int, int]]]):
    """Ensure all areas between squares are filled correctly."""
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 7:
                assert any(
                    min(s1[0], s2[0]) <= r <= max(s1[0], s2[0]) + 1 and
                    min(s1[1], s2[1]) <= c <= max(s1[1], s2[1]) + 1
                    for s1, s2 in generate_pairs(squares)
                ), f"Invalid orange cell at ({r}, {c})"
