from rob_agi.colored_grid import ColoredGrid

def solve_91714a58(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 91714a58 challenge by finding the largest valid rectangle of a single color.
    
    The function scans the input grid for each non-zero color, builds histograms of consecutive
    color heights, and uses a stack-based algorithm to find the largest rectangle in each histogram.
    It keeps track of the overall largest valid rectangle (at least 2x3 or 3x2) across all colors.
    Finally, it creates an output grid with the largest found rectangle filled with its color.

    The algorithm works as follows:
    1. Iterate through all non-zero colors in the input grid.
    2. For each color, scan the grid row by row, building a histogram of consecutive color heights.
    3. Use a stack-based algorithm to find the largest rectangle in each histogram.
    4. Keep track of the overall largest valid rectangle across all colors.
    5. Create an output grid of the same size as the input, filled with zeros.
    6. Fill the largest found rectangle in the output grid with its color.

    Args:
    input_grid (ColoredGrid): The input grid to be processed.

    Returns:
    ColoredGrid: The output grid with the largest valid rectangle filled.
    """
    def largest_rectangle(heights):
        stack = []
        max_area = 0
        max_rect = None
        heights.append(0)
        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                width = i - index
                area = width * height
                if area > max_area and ((width >= 3 and height >= 2) or (width >= 2 and height >= 3)):
                    max_area = area
                    max_rect = (index, i, height)
                start = index
            stack.append((start, h))
        heights.pop()
        return max_rect

    rows, cols = input_grid.get_dimensions()
    output = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    max_rect = None
    max_color = 0

    for color in range(1, 10):  # Assuming colors are 1-9
        heights = [0] * cols
        for r in range(rows):
            for c in range(cols):
                if input_grid.get_cell(r, c) == color:
                    heights[c] += 1
                else:
                    heights[c] = 0
            rect = largest_rectangle(heights)
            if rect:
                left, right, height = rect
                area = (right - left) * height
                if (not max_rect) or (area > (max_rect[2] - max_rect[1]) * max_rect[3]) or \
                   (area == (max_rect[2] - max_rect[1]) * max_rect[3] and 
                    (r - height + 1, left) < (max_rect[0], max_rect[1])):
                    max_rect = (r - height + 1, left, right, height)
                    max_color = color

    if max_rect:
        top, left, right, height = max_rect
        for r in range(top, top + height):
            for c in range(left, right):
                output.set_cell(r, c, max_color)

    return output
