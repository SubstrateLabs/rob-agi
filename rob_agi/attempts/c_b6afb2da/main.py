from rob_agi.colored_grid import ColoredGrid

def solve_b6afb2da(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by detecting rectangles of color 5 and replacing them with a pattern.
    The pattern consists of:
    - Corners (color 1)
    - Edges (color 4)
    - Interior (color 2)
    
    The function scans the grid, identifies rectangles, transforms them, and returns the modified grid.
    """
    def transform_rectangle(grid, top, left, bottom, right):
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if (r == top or r == bottom) and (c == left or c == right):
                    grid.set_cell(r, c, 1)  # corners
                elif r == top or r == bottom or c == left or c == right:
                    grid.set_cell(r, c, 4)  # edges
                else:
                    grid.set_cell(r, c, 2)  # interior

    output = input_grid.deep_copy()
    rows, cols = output.get_dimensions()

    for r in range(rows):
        c = 0
        while c < cols:
            if output.get_cell(r, c) == 5:
                # Find the bottom-right corner of the rectangle
                bottom, right = r, c
                while bottom + 1 < rows and output.get_cell(bottom + 1, c) == 5:
                    bottom += 1
                while right + 1 < cols and output.get_cell(r, right + 1) == 5:
                    right += 1
                
                # Transform the rectangle
                transform_rectangle(output, r, c, bottom, right)
                
                # Move to the next cell after this rectangle
                c = right + 1
            else:
                c += 1

    return output
