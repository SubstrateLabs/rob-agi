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

    # Place black squares
    black_positions = []
    for col in range(4):
        col_candidates = [pos for pos in candidates if pos[1] == col]
        if col_candidates:
            black_positions.append(col_candidates[0])
            candidates = [pos for pos in candidates if pos != col_candidates[0]]

    # Ensure each row has at least one black square and no more than two
    for row in range(rows):
        row_blacks = [pos for pos in black_positions if pos[0] == row]
        if not row_blacks:
            available_cols = [c for c in range(4) if not any(pos[1] == c for pos in black_positions)]
            if available_cols:
                new_pos = (row, available_cols[0])
                black_positions.append(new_pos)
                candidates = [pos for pos in candidates if pos != new_pos]
        elif len(row_blacks) == 1 and len(black_positions) < 6:
            available_cols = [c for c in range(4) if c != row_blacks[0][1] and not any(pos[1] == c for pos in black_positions)]
            if available_cols:
                new_pos = (row, available_cols[0])
                black_positions.append(new_pos)
                candidates = [pos for pos in candidates if pos != new_pos]

    # If we still don't have 6 black squares, add more
    while len(black_positions) < 6:
        for row in range(rows):
            if sum(1 for pos in black_positions if pos[0] == row) < 2:
                available_cols = [c for c in range(4) if not any(pos == (row, c) for pos in black_positions)]
                if available_cols:
                    new_pos = (row, available_cols[0])
                    black_positions.append(new_pos)
                    if len(black_positions) == 6:
                        break

    # Apply black squares to the output grid
    for r, c in black_positions:
        output_grid.set_cell(r, c, 0)

    return output_grid
