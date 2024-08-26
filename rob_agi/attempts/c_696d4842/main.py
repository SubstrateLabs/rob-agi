from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_696d4842(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by expanding shapes, connecting isolated cells, and applying color transformations.
    The process involves:
    1. Expanding shapes vertically and horizontally to their maximum extent.
    2. Connecting isolated cells to form larger shapes.
    3. Applying color transformations to the extremities of shapes.
    4. Filling gaps and removing remaining isolated cells.
    5. Iterating the process until no further changes are possible.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    color_cycle = {4: 2, 2: 6, 6: 1, 1: 3, 3: 8, 8: 4}  # Yellow -> Red -> Magenta -> Blue -> Green -> Sky -> Yellow

    def expand_shapes():
        for color in set(cell for row in output_grid.values for cell in row if cell != 0):
            cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == color]
            if len(cells) <= 1:
                continue

            # Vertical expansion
            for c in range(cols):
                color_cells = [r for r in range(rows) if output_grid.values[r][c] == color]
                if color_cells:
                    for r in range(min(color_cells), max(color_cells) + 1):
                        output_grid.values[r][c] = color

            # Horizontal expansion
            for r in range(rows):
                color_cells = [c for c in range(cols) if output_grid.values[r][c] == color]
                if color_cells:
                    for c in range(min(color_cells), max(color_cells) + 1):
                        output_grid.values[r][c] = color

    def connect_isolated_cells():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] != 0:
                    color = output_grid.values[r][c]
                    if all(output_grid.values[r+dr][c+dc] != color 
                           for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        # Try vertical connection
                        for dr in [-1, 1]:
                            new_r = r + dr
                            while 0 <= new_r < rows and output_grid.values[new_r][c] == 0:
                                output_grid.values[new_r][c] = color
                                new_r += dr
                        # If still isolated, try horizontal connection
                        if all(output_grid.values[r+dr][c+dc] != color 
                               for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                               if 0 <= r+dr < rows and 0 <= c+dc < cols):
                            for dc in [-1, 1]:
                                new_c = c + dc
                                while 0 <= new_c < cols and output_grid.values[r][new_c] == 0:
                                    output_grid.values[r][new_c] = color
                                    new_c += dc

    def transform_colors():
        for color in set(cell for row in output_grid.values for cell in row if cell != 0):
            cells = [(r, c) for r in range(rows) for c in range(cols) if output_grid.values[r][c] == color]
            if len(cells) <= 1:
                continue

            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)

            # Transform top
            for r in range(top, min(top + 3, bottom)):
                for c in range(cols):
                    if output_grid.values[r][c] == color:
                        output_grid.values[r][c] = color_cycle[color]

            # Transform bottom
            for r in range(max(bottom - 2, top + 3), bottom + 1):
                for c in range(cols):
                    if output_grid.values[r][c] == color:
                        output_grid.values[r][c] = color_cycle[color]

    def fill_gaps_and_remove_isolated():
        for r in range(rows):
            for c in range(cols):
                if output_grid.values[r][c] == 0:
                    neighbors = [output_grid.values[r+dr][c+dc] 
                                 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                 if 0 <= r+dr < rows and 0 <= c+dc < cols]
                    if len(set(neighbors) - {0}) == 1:
                        output_grid.values[r][c] = next(color for color in neighbors if color != 0)
                elif output_grid.values[r][c] != 0:
                    if all(output_grid.values[r+dr][c+dc] != output_grid.values[r][c] 
                           for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols):
                        output_grid.values[r][c] = 0

    iterations = 0
    while iterations < 10:  # Limit iterations to prevent infinite loop
        old_grid = output_grid.deep_copy()
        expand_shapes()
        connect_isolated_cells()
        transform_colors()
        fill_gaps_and_remove_isolated()
        if output_grid.values == old_grid.values:
            break
        iterations += 1

    return output_grid
