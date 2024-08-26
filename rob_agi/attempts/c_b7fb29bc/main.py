from rob_agi.colored_grid import ColoredGrid

def solve_b7fb29bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the area inside a green border with a complex pattern.
    
    The pattern consists of:
    1. A yellow (4) border just inside the green (3) border
    2. A complex pattern of red (2) and yellow (4) in the interior
    3. A gradient from more red in the top-left to more yellow in the bottom-right
    4. Preservation of any original green (3) cells within the border
    5. Special handling for small interiors
    6. The bottom row is always red (2), except in small interiors
    7. Alternating columns of red and yellow in the top rows
    """
    # Step 1: Identify the green border
    rows, cols = input_grid.get_dimensions()
    top = next(r for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) == 3)
    left = next(c for c in range(cols) if input_grid.get_cell(top, c) == 3)
    bottom = next(r for r in range(rows-1, -1, -1) for c in range(cols) if input_grid.get_cell(r, c) == 3)
    right = next(c for c in range(cols-1, -1, -1) if input_grid.get_cell(top, c) == 3)

    # Step 2: Create a deep copy of the input grid
    result = input_grid.deep_copy()

    # Step 3: Calculate dimensions of the inner area
    inner_top, inner_left = top + 1, left + 1
    inner_bottom, inner_right = bottom - 1, right - 1
    inner_height = inner_bottom - inner_top + 1
    inner_width = inner_right - inner_left + 1

    # Step 4: Fill the interior with the complex pattern
    for r in range(inner_top, inner_bottom + 1):
        for c in range(inner_left, inner_right + 1):
            if r == inner_top or r == inner_bottom or c == inner_left or c == inner_right:
                result.set_cell(r, c, 4)  # Yellow border
            elif r == inner_bottom:
                result.set_cell(r, c, 2)  # Bottom row is red
            else:
                # Create a gradient pattern
                x = (c - inner_left) / (inner_width - 1)
                y = (r - inner_top) / (inner_height - 1)
                if y < 0.5:  # Top half
                    result.set_cell(r, c, 2 if c % 2 == 0 else 4)  # Alternating red and yellow
                elif x < 0.5:  # Bottom-left quadrant
                    result.set_cell(r, c, 2)  # Red
                else:  # Bottom-right quadrant
                    result.set_cell(r, c, 4)  # Yellow

    # Step 5: Handle small interiors
    if inner_width <= 3 or inner_height <= 3:
        for r in range(inner_top, inner_bottom + 1):
            for c in range(inner_left, inner_right + 1):
                result.set_cell(r, c, 4)  # Fill with yellow for small interiors

    # Step 6: Preserve original green cells
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if input_grid.get_cell(r, c) == 3:
                result.set_cell(r, c, 3)

    return result
