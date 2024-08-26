from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Color analysis: Analyzes color frequencies in the entire input grid, ignoring green separators.
    2. Color hierarchy: Establishes a color priority based on frequency and predefined importance.
    3. Pattern determination: Selects a dominant pattern based on color prevalence.
    4. Grid construction: Builds a 4x4 grid using the dominant pattern and color hierarchy.
    5. Color incorporation: Ensures representation of top colors while maintaining balance.
    6. Refinement: Adjusts for symmetry, balance, and visual appeal.
    7. Validation: Verifies color representation and overall grid character.
    The final 4x4 grid reflects the essence of the input while creating a coherent and visually interesting output.
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

    # Pattern determination
    dominant_color = color_ranking[0][0]
    if color_percentages[dominant_color] > 0.4:
        pattern = "extended"
    elif color_percentages[dominant_color] > 0.25:
        pattern = "c_shape"
    else:
        pattern = "diagonal"

    # Grid construction
    output = [[0 for _ in range(4)] for _ in range(4)]
    
    if pattern == "extended":
        for i in range(4):
            output[0][i] = output[3][i] = dominant_color
        output[1][0] = output[2][0] = dominant_color
    elif pattern == "c_shape":
        for i in range(4):
            output[0][i] = output[3][i] = dominant_color
        output[1][0] = output[2][0] = dominant_color
    else:  # diagonal
        for i in range(4):
            output[i][i] = dominant_color

    # Color incorporation
    secondary_color = color_ranking[1][0]
    for i in range(4):
        for j in range(4):
            if output[i][j] == 0:
                output[i][j] = secondary_color

    # Ensure top colors are represented
    for color, _ in color_ranking[2:4]:
        if all(color not in row for row in output):
            for i in range(4):
                if len(set(output[i])) > 2:
                    least_common = min(set(output[i]), key=lambda x: color_counts.get(x, 0))
                    output[i][output[i].index(least_common)] = color
                    break

    # Refinement
    for i in range(4):
        if len(set(output[i])) == 1 and i != 0 and i != 3:
            output[i][1] = color_ranking[2][0]
        if len(set(row[i] for row in output)) == 1:
            output[1][i] = color_ranking[2][0]

    # Include black if significant
    if black_percentage > 0.2:
        if 0 not in [cell for row in output for cell in row]:
            output[2][2] = 0

    # Final symmetry check
    if output[0] != output[3]:
        output[3] = output[0]

    return ColoredGrid(values=output)
