from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_c97c0139(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c97c0139 challenge by creating diamond-shaped sky blue (8) fields around red (2) lines.
    
    The solution follows these steps:
    1. Identify red lines in the input grid.
    2. Generate a diamond-shaped field around each red line.
    3. Combine all fields and apply them to the grid, setting non-red cells to sky blue.
    
    This approach works for both horizontal and vertical red lines of varying lengths.
    """
    def find_red_lines(grid: ColoredGrid) -> List[Tuple[int, int, int, int]]:
        lines = []
        rows, cols = grid.get_dimensions()
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 2:  # Red cell
                    if not lines or (r, c) != (lines[-1][2], lines[-1][3] + 1):  # New line
                        lines.append((r, c, r, c))
                    else:  # Extend existing line
                        lines[-1] = (lines[-1][0], lines[-1][1], r, c)
        return lines

    def generate_field(line: Tuple[int, int, int, int], rows: int, cols: int) -> Set[Tuple[int, int]]:
        field = set()
        start_r, start_c, end_r, end_c = line
        is_horizontal = start_r == end_r
        line_length = max(end_c - start_c, end_r - start_r) + 1
        max_distance = (line_length + 1) // 2

        for r in range(max(0, start_r - max_distance), min(rows, end_r + max_distance + 1)):
            for c in range(max(0, start_c - max_distance), min(cols, end_c + max_distance + 1)):
                if is_horizontal:
                    distance = abs(r - start_r) + min(abs(c - start_c), abs(c - end_c))
                else:
                    distance = abs(c - start_c) + min(abs(r - start_r), abs(r - end_r))
                if distance < max_distance:
                    field.add((r, c))

        return field

    # Find red lines
    red_lines = find_red_lines(input_grid)

    # Generate and combine fields
    rows, cols = input_grid.get_dimensions()
    combined_field = set()
    for line in red_lines:
        combined_field.update(generate_field(line, rows, cols))

    # Create output grid
    output_grid = input_grid.deep_copy()
    for r, c in combined_field:
        if output_grid.get_cell(r, c) != 2:  # If not red
            output_grid.set_cell(r, c, 8)  # Set to sky blue

    return output_grid
