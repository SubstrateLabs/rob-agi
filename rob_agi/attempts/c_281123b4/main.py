from rob_agi.colored_grid import ColoredGrid
from collections import Counter
import random

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Input Analysis: Analyzes color frequencies and distribution in the input grid.
    2. Color Palette Creation: Selects the most prominent colors for the output.
    3. Output Grid Construction: Builds a 4x4 grid using the color palette.
    4. Pattern Application: Applies specific patterns based on color dominance.
    5. Color Distribution: Distributes remaining colors based on input frequencies.
    6. Balance and Refinement: Adjusts the grid for visual interest and color representation.
    7. Final Validation: Ensures all palette colors are represented in the output.
    The final 4x4 grid reflects the essence of the input while creating a visually coherent output.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    # Input Analysis
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell not in [0, 3])
    total_colored_cells = sum(color_counts.values())
    color_percentages = {color: count / total_colored_cells for color, count in color_counts.items()}
    black_percentage = sum(row.count(0) for row in input_grid.values) / (rows * cols)

    # Color Palette Creation
    palette = sorted(color_counts, key=color_counts.get, reverse=True)[:4]
    if black_percentage > 0.15 and 0 not in palette:
        palette[-1] = 0

    # Output Grid Construction
    output = [[0 for _ in range(4)] for _ in range(4)]
    dominant_color = palette[0]

    # Pattern Application
    if color_percentages[dominant_color] > 0.4:
        for i in range(4):
            output[i][0] = output[i][-1] = dominant_color
        output[0][1:3] = [dominant_color, dominant_color]
    else:
        for i in range(4):
            output[i][i] = output[i][3-i] = dominant_color

    # Color Distribution
    remaining_cells = [(r, c) for r in range(4) for c in range(4) if output[r][c] == 0]
    for color in palette[1:]:
        cells_to_fill = int(len(remaining_cells) * color_percentages.get(color, 0) / sum(color_percentages.values()))
        for _ in range(cells_to_fill):
            if remaining_cells:
                r, c = remaining_cells.pop(0)
                output[r][c] = color

    # Fill remaining cells
    for r, c in remaining_cells:
        output[r][c] = random.choice(palette)

    # Balance and Refinement
    for _ in range(2):
        for r in range(4):
            for c in range(4):
                neighbors = [output[nr][nc] for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                             if 0 <= nr < 4 and 0 <= nc < 4]
                if len(set(neighbors)) < 2:
                    output[r][c] = random.choice([color for color in palette if color != output[r][c]])

    # Final Validation
    for color in palette:
        if all(color not in row for row in output):
            r, c = random.choice([(r, c) for r in range(4) for c in range(4)])
            output[r][c] = color

    return ColoredGrid(values=output)
