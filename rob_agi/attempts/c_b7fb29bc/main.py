from rob_agi.colored_grid import ColoredGrid
import random

def solve_b7fb29bc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling the area inside a green border with a complex pattern.
    
    The pattern consists of:
    1. A yellow (4) border just inside the green (3) border
    2. A complex pattern of red (2) and yellow (4) in the interior, based on:
       - Position of the original green cell (if any)
       - Distance from the center or original green cell
       - Horizontal and vertical striping
    3. Preservation of any original green (3) cells within the border
    4. Special handling for small interiors
    5. Balanced distribution of red and yellow (approx. 40% red, 60% yellow)
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

    # Step 4: Find the original green cell (if any)
    original_green = None
    for r in range(inner_top, inner_bottom + 1):
        for c in range(inner_left, inner_right + 1):
            if input_grid.get_cell(r, c) == 3:
                original_green = (r, c)
                break
        if original_green:
            break

    # Step 5: Determine pattern orientation
    use_horizontal = original_green is None or original_green[0] < (inner_top + inner_bottom) // 2

    # Step 6: Fill the interior with the complex pattern
    center = ((inner_top + inner_bottom) // 2, (inner_left + inner_right) // 2)
    max_distance = max(inner_height, inner_width) // 2

    for r in range(inner_top, inner_bottom + 1):
        for c in range(inner_left, inner_right + 1):
            if r == inner_top or r == inner_bottom or c == inner_left or c == inner_right:
                result.set_cell(r, c, 4)  # Yellow border
            elif input_grid.get_cell(r, c) == 3:
                continue  # Preserve original green cell
            else:
                # Calculate distance factor
                if original_green:
                    distance = max(abs(r - original_green[0]), abs(c - original_green[1]))
                else:
                    distance = max(abs(r - center[0]), abs(c - center[1]))
                distance_factor = 1 - (distance / max_distance)

                # Apply striping
                stripe_factor = 0.2 if (r + c) % 2 == 0 else -0.2

                # Determine color probability
                yellow_prob = 0.6 + distance_factor * 0.2 + stripe_factor

                if use_horizontal:
                    if r == inner_top + 1:
                        result.set_cell(r, c, 4 if c % 2 == 0 else 2)
                    elif r == inner_bottom:
                        result.set_cell(r, c, 4 if original_green and original_green[0] > center[0] else 2)
                    else:
                        result.set_cell(r, c, 4 if random.random() < yellow_prob else 2)
                else:
                    if r <= center[0]:
                        result.set_cell(r, c, 4 if random.random() < yellow_prob else 2)
                    else:
                        result.set_cell(r, c, 4 if c % 2 == 0 else 2)

    # Step 7: Handle small interiors
    if inner_width <= 5 or inner_height <= 5:
        for r in range(inner_top, inner_bottom + 1):
            for c in range(inner_left, inner_right + 1):
                if input_grid.get_cell(r, c) != 3:
                    result.set_cell(r, c, 4 if (r + c) % 2 == 0 else 2)

    # Step 8: Adjust around the original green cell
    if original_green:
        r, c = original_green
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if inner_top <= nr <= inner_bottom and inner_left <= nc <= inner_right:
                result.set_cell(nr, nc, 4)

    # Step 9: Final pass to balance colors
    yellow_count = sum(1 for r in range(inner_top, inner_bottom + 1)
                       for c in range(inner_left, inner_right + 1)
                       if result.get_cell(r, c) == 4)
    total_cells = (inner_bottom - inner_top + 1) * (inner_right - inner_left + 1)
    yellow_ratio = yellow_count / total_cells

    if yellow_ratio < 0.55:
        for r in range(inner_top, inner_bottom + 1):
            for c in range(inner_left, inner_right + 1):
                if result.get_cell(r, c) == 2 and random.random() < 0.2:
                    result.set_cell(r, c, 4)
    elif yellow_ratio > 0.65:
        for r in range(inner_top, inner_bottom + 1):
            for c in range(inner_left, inner_right + 1):
                if result.get_cell(r, c) == 4 and random.random() < 0.2:
                    result.set_cell(r, c, 2)

    return result
