from rob_agi.colored_grid import ColoredGrid
import math

def solve_85b81ff1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by redistributing black cells (0's) 
    in non-black columns while maintaining the structure at the top and bottom.
    
    The solution follows these steps:
    1. Copy the input grid.
    2. For each non-black column:
       a. Identify unchangeable top and bottom sections.
       b. In the middle section, redistribute black cells evenly.
    3. Return the modified grid.
    
    This approach balances the distribution of gaps in the colored pillars
    while preserving the overall structure of the grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()

    for col in range(cols):
        if all(input_grid.get_cell(row, col) == 0 for row in range(rows)):
            continue  # Skip all-black columns

        # Find the first and last black cell
        first_black = next(row for row in range(rows) if input_grid.get_cell(row, col) == 0)
        last_black = next(row for row in range(rows-1, -1, -1) if input_grid.get_cell(row, col) == 0)

        # Define the middle section
        middle_start = first_black + 2
        middle_end = last_black - 1

        if middle_end <= middle_start:
            continue  # No middle section to process

        # Count black cells in the middle section
        black_cells = sum(1 for row in range(middle_start, middle_end+1) if input_grid.get_cell(row, col) == 0)
        total_cells = middle_end - middle_start + 1

        if black_cells == 0:
            continue  # No black cells to redistribute

        # Redistribute black cells
        ideal_gap = math.floor(total_cells / (black_cells + 1))
        current_row = middle_start
        for _ in range(black_cells):
            for _ in range(ideal_gap):
                if current_row <= middle_end:
                    output_grid.set_cell(current_row, col, input_grid.get_cell(current_row, col))
                    current_row += 1
            if current_row <= middle_end:
                output_grid.set_cell(current_row, col, 0)
                current_row += 1

        # Fill the rest with colored cells
        while current_row <= middle_end:
            output_grid.set_cell(current_row, col, input_grid.get_cell(current_row, col))
            current_row += 1

    return output_grid
