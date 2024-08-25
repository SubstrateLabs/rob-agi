from rob_agi.colored_grid import ColoredGrid
import random

def solve_5d2a5c43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by following these steps:
    1. Extract the left 4 columns from the input grid.
    2. Analyze the distribution of black cells in the input.
    3. Create a new 6x4 grid filled with sky blue (8).
    4. Distribute black cells (0) in the output grid based on the input pattern.
    5. Balance the distribution and ensure no row is entirely black.
    6. Return the transformed 6x4 output grid.
    """
    rows, cols = input_grid.get_dimensions()
    extracted_grid = input_grid.extract_subgrid(0, 0, rows, 4)
    output_grid = ColoredGrid(values=[[8 for _ in range(4)] for _ in range(rows)])

    # Analyze input
    input_black_count = sum(extracted_grid.get_cell(r, c) == 0 for r in range(rows) for c in range(4))
    target_black_count = int(input_black_count * 0.7)  # 70% of input black cells

    # Distribute black cells
    placed_black_cells = 0
    for r in range(rows):
        row_black_count = 0
        for c in range(4):
            if placed_black_cells < target_black_count and random.random() < 0.4:  # 40% chance to place a black cell
                if row_black_count < 2:  # Ensure no more than 2 black cells per row
                    output_grid.set_cell(r, c, 0)
                    placed_black_cells += 1
                    row_black_count += 1

    # Balance distribution
    for r in range(rows):
        row_black_count = sum(output_grid.get_cell(r, c) == 0 for c in range(4))
        if row_black_count == 0:
            c = random.randint(0, 3)
            output_grid.set_cell(r, c, 0)
            placed_black_cells += 1
        elif row_black_count > 2:
            black_cells = [c for c in range(4) if output_grid.get_cell(r, c) == 0]
            cells_to_remove = random.sample(black_cells, row_black_count - 2)
            for c in cells_to_remove:
                output_grid.set_cell(r, c, 8)
                placed_black_cells -= 1

    # Ensure no all-black rows
    for r in range(rows):
        if all(output_grid.get_cell(r, c) == 0 for c in range(4)):
            output_grid.set_cell(r, 3, 8)
            placed_black_cells -= 1

    # Final adjustment to match target black count
    while placed_black_cells < target_black_count:
        r, c = random.randint(0, rows-1), random.randint(0, 3)
        if output_grid.get_cell(r, c) == 8:
            output_grid.set_cell(r, c, 0)
            placed_black_cells += 1

    return output_grid
