from rob_agi.colored_grid import ColoredGrid

def solve_29c11459(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Only rows with non-zero values at the ends are modified.
    2. The leftmost non-zero value is propagated to the left half of the row.
    3. The rightmost non-zero value is propagated to the right half of the row.
    4. The middle cell is always set to 5 (gray) for modified rows.
    """
    def transform_row(row):
        if row[0] == 0 and row[-1] == 0:
            return row
        left_color = next(color for color in row if color != 0)
        right_color = next(color for color in reversed(row) if color != 0)
        mid = len(row) // 2
        return [left_color] * mid + [5] + [right_color] * (len(row) - mid - 1)

    height, width = input_grid.get_dimensions()
    transformed_values = [transform_row(input_grid.values[i]) for i in range(height)]
    return ColoredGrid(values=transformed_values)
