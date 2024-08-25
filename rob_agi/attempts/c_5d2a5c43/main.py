from rob_agi.colored_grid import ColoredGrid

def solve_5d2a5c43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by following these steps:
    1. Extract the left 4 columns from the input grid.
    2. Create a new 6x4 grid filled with sky blue (8).
    3. Process each column of the extracted grid to place black cells (0) in the output grid.
    4. Balance the distribution of black cells and preserve key aspects of the input pattern.
    5. Return the transformed 6x4 output grid.
    """
    rows, cols = input_grid.get_dimensions()
    extracted_grid = input_grid.extract_subgrid(0, 0, rows, 4)
    output_grid = ColoredGrid(values=[[8 for _ in range(4)] for _ in range(rows)])

    # Process first two columns
    for r in range(rows):
        for c in range(2):
            if extracted_grid.get_cell(r, c) == 0:
                output_grid.set_cell(r, c, 0)

    # Process third and fourth columns
    for r in range(rows):
        black_count = sum(1 for c in range(2) if output_grid.get_cell(r, c) == 0)
        for c in range(2, 4):
            if extracted_grid.get_cell(r, c) == 0 and black_count < 2:
                output_grid.set_cell(r, 3 if c == 2 else 2, 0)
                black_count += 1

    # Balance check and adjustment
    left_black = sum(output_grid.get_cell(r, c) == 0 for r in range(rows) for c in range(2))
    right_black = sum(output_grid.get_cell(r, c) == 0 for r in range(rows) for c in range(2, 4))
    
    if abs(left_black - right_black) > 1:
        for r in range(rows):
            if left_black > right_black and output_grid.get_cell(r, 1) == 0 and output_grid.get_cell(r, 2) == 8:
                output_grid.set_cell(r, 1, 8)
                output_grid.set_cell(r, 2, 0)
                left_black -= 1
                right_black += 1
            elif right_black > left_black and output_grid.get_cell(r, 2) == 0 and output_grid.get_cell(r, 1) == 8:
                output_grid.set_cell(r, 2, 8)
                output_grid.set_cell(r, 1, 0)
                right_black -= 1
                left_black += 1
            if abs(left_black - right_black) <= 1:
                break

    # Ensure no all-black rows
    for r in range(rows):
        if all(output_grid.get_cell(r, c) == 0 for c in range(4)):
            output_grid.set_cell(r, 3, 8)

    return output_grid
