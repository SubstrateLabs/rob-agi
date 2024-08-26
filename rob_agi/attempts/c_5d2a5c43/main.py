from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5d2a5c43(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by following these steps:
    1. Extract the left 4 columns from the input grid.
    2. Create a new 6x4 grid filled with sky blue (8).
    3. Identify candidate positions for black squares based on non-yellow cells in the input.
    4. Place exactly 6 black squares in the output grid:
       - One in each column
       - No more than two in any row
       - At least one in each row
    5. Return the transformed 6x4 output grid.
    """
    rows, cols = input_grid.get_dimensions()
    extracted_grid = input_grid.extract_subgrid(0, 0, rows, 4)
    output_grid = ColoredGrid(values=[[8 for _ in range(4)] for _ in range(rows)])

    # Create a list of candidate positions for black squares
    candidates = [(r, c) for r in range(rows) for c in range(4) if extracted_grid.get_cell(r, c) != 4]

    # Ensure we have at least 6 candidates
    while len(candidates) < 6:
        r, c = divmod(len(candidates), 4)
        candidates.append((r, c))

    # Sort candidates by column, then by row
    candidates.sort(key=lambda x: (x[1], x[0]))

    # Select black square positions
    black_positions = []
    for col in range(4):
        col_candidates = [pos for pos in candidates if pos[1] == col]
        if col_candidates:
            black_positions.append(col_candidates[0])
            candidates = [pos for pos in candidates if pos != col_candidates[0]]

    # Add additional black squares
    while len(black_positions) < 6:
        for row in range(rows):
            row_count = sum(1 for pos in black_positions if pos[0] == row)
            if row_count < 2:
                available_cols = [c for c in range(4) if not any(pos[1] == c for pos in black_positions if pos[0] == row)]
                if available_cols:
                    black_positions.append((row, available_cols[0]))
                    if len(black_positions) == 6:
                        break

    # Ensure each row has at least one black square
    for row in range(rows):
        if not any(pos[0] == row for pos in black_positions):
            available_cols = [c for c in range(4) if not any(pos[1] == c for pos in black_positions)]
            if available_cols:
                black_positions.append((row, available_cols[0]))

    # Apply black squares to the output grid
    for r, c in black_positions:
        output_grid.set_cell(r, c, 0)

    return output_grid
