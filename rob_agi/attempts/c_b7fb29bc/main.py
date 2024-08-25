from rob_agi.colored_grid import ColoredGrid

def solve_b7fb29bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the area inside a green border with a complex pattern.
    
    The pattern consists of:
    1. A yellow (4) border just inside the green (3) border
    2. Three horizontal sections with specific red (2) and yellow (4) patterns
    3. A central vertical yellow (4) line
    4. Preservation of any original green (3) cells within the border
    
    The top section has alternating yellow and red vertical stripes, starting with yellow.
    The middle section is yellow with red on the left and right edges.
    The bottom section has alternating red and yellow vertical stripes, starting with red.
    """
    # Step 1: Identify the green border
    rows, cols = input_grid.get_dimensions()
    top = next(r for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 3)
    left = next(c for c in range(cols) if input_grid.get_cell(top, c) == 3)
    bottom = next(r for r in range(rows-1, -1, -1) for c in range(cols) if input_grid.get_cell(r, c) == 3)
    right = next(c for c in range(cols-1, -1, -1) if input_grid.get_cell(top, c) == 3)

    # Step 2: Create a deep copy of the input grid
    result = input_grid.deep_copy()

    # Step 3: Create a yellow border just inside the green border
    for r in range(top+1, bottom):
        for c in range(left+1, right):
            if input_grid.get_cell(r, c) != 3:
                result.set_cell(r, c, 4)

    # Step 4: Calculate dimensions of the inner area
    inner_top, inner_left = top + 2, left + 2
    inner_bottom, inner_right = bottom - 2, right - 2
    inner_height = inner_bottom - inner_top + 1
    inner_width = inner_right - inner_left + 1

    # Step 5: Determine the central vertical line
    mid_col = (inner_left + inner_right) // 2

    # Step 6: Divide the inner area into three horizontal sections
    section_height = inner_height // 3
    extra_rows = inner_height % 3

    # Step 7: Fill the top section
    for r in range(inner_top, inner_top + section_height + (1 if extra_rows > 0 else 0)):
        for c in range(inner_left, inner_right + 1):
            if c == mid_col or (c - inner_left) % 2 == 0:
                result.set_cell(r, c, 4)
            else:
                result.set_cell(r, c, 2)

    # Step 8: Fill the middle section
    for r in range(inner_top + section_height + (1 if extra_rows > 0 else 0), inner_bottom - section_height + (1 if extra_rows == 2 else 0)):
        for c in range(inner_left, inner_right + 1):
            if c == inner_left or c == inner_right:
                result.set_cell(r, c, 2)
            else:
                result.set_cell(r, c, 4)

    # Step 9: Fill the bottom section
    for r in range(inner_bottom - section_height + (1 if extra_rows == 2 else 0), inner_bottom + 1):
        for c in range(inner_left, inner_right + 1):
            if c == mid_col or (c - inner_left) % 2 == 1:
                result.set_cell(r, c, 4)
            else:
                result.set_cell(r, c, 2)

    # Step 10: Preserve original green cells
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if input_grid.get_cell(r, c) == 3:
                result.set_cell(r, c, 3)

    # Step 11: Final check
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if result.get_cell(r, c) == 0:
                result.set_cell(r, c, 4)

    return result
