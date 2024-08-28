from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_05a7bcf2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid according to the following rules:
    1. Analyzes the input grid to determine color positions and center of mass.
    2. Identifies or creates a sky blue (8) barrier based on color alignments and center of mass.
    3. Expands yellow (4) regions downwards/rightwards until hitting the barrier, red, or edge.
    4. Expands red (2) regions upwards/leftwards until hitting the barrier, yellow, or edge.
    5. Fills the top/left section with green (3) where not yellow or sky blue.
    6. Fills remaining cells in bottom/right with sky blue (8).
    7. Ensures the integrity of the sky blue barrier.
    8. Respects original positions of colors during expansion.

    Args:
    input_grid (ColoredGrid): The input grid to transform.

    Returns:
    ColoredGrid: The transformed grid.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def calculate_center_of_mass(grid):
        total_mass = 0
        sum_x, sum_y = 0, 0
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] != 0:
                    total_mass += 1
                    sum_x += c
                    sum_y += r
        if total_mass == 0:
            return rows // 2, cols // 2
        return sum_y // total_mass, sum_x // total_mass

    def find_sky_blue_barrier(grid):
        horizontal_count = max(sum(1 for cell in row if cell == 8) for row in grid.values)
        vertical_count = max(sum(1 for row in grid.values if row[col] == 8) for col in range(cols))
        
        center_y, center_x = calculate_center_of_mass(grid)
        
        if horizontal_count > vertical_count:
            return ('horizontal', center_y)
        elif vertical_count > horizontal_count:
            return ('vertical', center_x)
        else:
            return ('horizontal', center_y)  # Default to horizontal if counts are equal

    def complete_barrier(grid, orientation, pos):
        if orientation == 'horizontal':
            grid.values[pos] = [8] * cols
        else:
            for r in range(rows):
                grid.values[r][pos] = 8

    def expand_color(grid, start_row, start_col, color, direction, original_positions):
        if direction == 'down':
            for r in range(start_row, rows):
                if grid.values[r][start_col] in [8, 2] or (r, start_col) in original_positions:
                    break
                grid.values[r][start_col] = color
        elif direction == 'up':
            for r in range(start_row, -1, -1):
                if grid.values[r][start_col] in [8, 4] or (r, start_col) in original_positions:
                    break
                grid.values[r][start_col] = color
        elif direction == 'right':
            for c in range(start_col, cols):
                if grid.values[start_row][c] in [8, 2] or (start_row, c) in original_positions:
                    break
                grid.values[start_row][c] = color
        elif direction == 'left':
            for c in range(start_col, -1, -1):
                if grid.values[start_row][c] in [8, 4] or (start_row, c) in original_positions:
                    break
                grid.values[start_row][c] = color

    orientation, barrier_pos = find_sky_blue_barrier(grid)
    complete_barrier(grid, orientation, barrier_pos)

    # Store original color positions
    original_positions = set((r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] in [2, 4, 8])

    if orientation == 'horizontal':
        # Expand yellow downwards and rightwards
        for r in range(barrier_pos):
            for c in range(cols):
                if grid.values[r][c] == 4:
                    expand_color(grid, r, c, 4, 'down', original_positions)
                    expand_color(grid, r, c, 4, 'right', original_positions)
        
        # Expand red upwards and leftwards
        for r in range(barrier_pos + 1, rows):
            for c in range(cols - 1, -1, -1):
                if grid.values[r][c] == 2:
                    expand_color(grid, r, c, 2, 'up', original_positions)
                    expand_color(grid, r, c, 2, 'left', original_positions)
        
        # Fill with green in top half
        for r in range(barrier_pos):
            for c in range(cols):
                if grid.values[r][c] not in [4, 8]:
                    grid.values[r][c] = 3
        
        # Fill remaining cells with sky blue
        for r in range(barrier_pos + 1, rows):
            for c in range(cols):
                if grid.values[r][c] not in [2, 8]:
                    grid.values[r][c] = 8
    else:  # vertical orientation
        # Expand yellow downwards and rightwards
        for r in range(rows):
            for c in range(barrier_pos):
                if grid.values[r][c] == 4:
                    expand_color(grid, r, c, 4, 'down', original_positions)
                    expand_color(grid, r, c, 4, 'right', original_positions)
        
        # Expand red upwards and leftwards
        for r in range(rows - 1, -1, -1):
            for c in range(barrier_pos + 1, cols):
                if grid.values[r][c] == 2:
                    expand_color(grid, r, c, 2, 'up', original_positions)
                    expand_color(grid, r, c, 2, 'left', original_positions)
        
        # Fill with green in left half
        for c in range(barrier_pos):
            for r in range(rows):
                if grid.values[r][c] not in [4, 8]:
                    grid.values[r][c] = 3
        
        # Fill remaining cells with sky blue
        for c in range(barrier_pos + 1, cols):
            for r in range(rows):
                if grid.values[r][c] not in [2, 8]:
                    grid.values[r][c] = 8

    # Final check to ensure barrier integrity
    complete_barrier(grid, orientation, barrier_pos)

    return grid
