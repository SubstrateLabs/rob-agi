from rob_agi.colored_grid import ColoredGrid
from collections import Counter
import random

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Color analysis: Analyzes color frequencies in the input grid, excluding green separators.
    2. Color hierarchy: Establishes a color priority based on frequency and predefined importance.
    3. Palette creation: Selects top colors to form the output palette.
    4. Grid construction: Builds a 4x4 grid using the color palette and various patterns.
    5. Balance and refinement: Adjusts the grid for visual interest and color representation.
    6. Validation: Ensures the output captures the essence of the input while maintaining flexibility.
    The final 4x4 grid reflects the character of the input while creating a visually interesting output.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    # Color analysis
    color_counts = Counter(cell for row in input_grid.values for cell in row if cell not in [0, 3])
    total_colored_cells = sum(color_counts.values())
    color_percentages = {color: count / total_colored_cells for color, count in color_counts.items()}
    black_percentage = sum(row.count(0) for row in input_grid.values) / (rows * cols)

    # Color hierarchy
    color_weights = {9: 4, 4: 3, 8: 2, 5: 1}  # Brown, Yellow, Sky, Gray
    color_ranking = sorted(color_counts.items(), key=lambda x: (color_weights.get(x[0], 0), x[1]), reverse=True)

    # Palette creation
    palette = [color for color, _ in color_ranking[:3]]
    if black_percentage > 0.15:
        palette.append(0)
    if len(palette) < 4:
        palette.append(color_ranking[3][0])

    # Grid construction
    output = [[0 for _ in range(4)] for _ in range(4)]
    dominant_color = palette[0]
    
    # Apply dominant color pattern
    if color_percentages[dominant_color] > 0.4:
        for i in range(4):
            output[0][i] = output[3][i] = dominant_color
        output[1][0] = output[2][0] = dominant_color
    else:
        for i in range(4):
            output[i][i] = dominant_color
            output[i][3-i] = dominant_color

    # Distribute other colors
    remaining_cells = [(r, c) for r in range(4) for c in range(4) if output[r][c] == 0]
    random.shuffle(remaining_cells)
    for i, color in enumerate(palette[1:]):
        cells_to_fill = len(remaining_cells) // (len(palette) - 1)
        for _ in range(cells_to_fill):
            if remaining_cells:
                r, c = remaining_cells.pop()
                output[r][c] = color

    # Fill any remaining cells
    for r, c in remaining_cells:
        output[r][c] = random.choice(palette)

    # Balance and refinement
    for _ in range(2):
        for r in range(4):
            for c in range(4):
                neighbors = [output[nr][nc] for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                             if 0 <= nr < 4 and 0 <= nc < 4]
                if all(output[r][c] == neighbor for neighbor in neighbors):
                    output[r][c] = random.choice([color for color in palette if color != output[r][c]])

    # Ensure all palette colors are represented
    for color in palette:
        if all(color not in row for row in output):
            r, c = random.choice([(r, c) for r in range(4) for c in range(4)])
            output[r][c] = color

    return ColoredGrid(values=output)
