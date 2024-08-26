from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_62ab2642(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by applying the following rules:
    1. Fill the largest contiguous area of black (0) cells with sky blue (8), starting from the top-right or bottom-right corner.
    2. Identify small areas of black cells adjacent to gray (5) cells and fill them with orange (7).
    3. Preserve all original gray (5) cells.

    The function uses flood fill for the sky blue area and a scanning approach for orange areas.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def flood_fill(x: int, y: int, target_color: int, replacement_color: int) -> int:
        if (x < 0 or x >= cols or y < 0 or y >= rows or 
            output_grid.values[y][x] != target_color):
            return 0
        
        output_grid.values[y][x] = replacement_color
        count = 1
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            count += flood_fill(x + dx, y + dy, target_color, replacement_color)
        return count

    # Fill sky blue area
    sky_blue_count = flood_fill(cols - 1, 0, 0, 8)
    if sky_blue_count < 10:  # If top-right fill is small, try bottom-right
        output_grid = input_grid.deep_copy()  # Reset grid
        flood_fill(cols - 1, rows - 1, 0, 8)

    # Identify potential orange areas
    potential_orange: Set[Tuple[int, int]] = set()
    for y in range(rows):
        for x in range(cols):
            if output_grid.values[y][x] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < cols and 0 <= ny < rows and
                            output_grid.values[ny][nx] == 5):
                            potential_orange.add((x, y))
                            break
                    if (x, y) in potential_orange:
                        break

    # Apply orange filling
    for x, y in potential_orange:
        if output_grid.values[y][x] == 0:
            output_grid.values[y][x] = 7
            # Check adjacent cell
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if ((nx, ny) in potential_orange and
                    0 <= nx < cols and 0 <= ny < rows and
                    output_grid.values[ny][nx] == 0):
                    output_grid.values[ny][nx] = 7

    return output_grid
