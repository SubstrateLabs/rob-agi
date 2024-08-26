from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Section division: Divides each row into 4 sections, ignoring the green (3) separator columns.
    2. Color analysis: Analyzes color frequencies and patterns in each section and globally.
    3. Output construction: Builds a 4x4 grid based on color priorities, patterns, and global color distribution.
    4. Pattern enhancement: Creates diagonal patterns and ensures color balance.
    5. Refinement: Adjusts the output to create balanced, symmetric patterns and preserve key characteristics of the input.
    The final 4x4 grid reflects the most prominent colors and patterns from the input while maintaining coherence and aesthetic appeal.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    def analyze_section(section: List[int]) -> Tuple[int, int, Counter]:
        color_counts = Counter(color for color in section if color != 0 and color != 3)
        if not color_counts:
            return 0, 0, color_counts
        primary = max(color_counts.items(), key=lambda x: (x[1], x[0]))[0]
        secondary = max((c for c in color_counts if c != primary), default=0, key=lambda x: (color_counts[x], x))
        return primary, secondary, color_counts

    # Section analysis
    sections = []
    global_counts = Counter()
    for row in input_grid.values:
        row_sections = [row[0:5], row[5:10], row[10:15], row[15:19]]
        row_analysis = [analyze_section(section) for section in row_sections]
        sections.append(row_analysis)
        for _, _, counts in row_analysis:
            global_counts.update(counts)

    # Global color ranking
    color_weights = {9: 4, 4: 3, 8: 2, 5: 1}  # Brown, Yellow, Sky, Gray
    global_ranking = sorted(global_counts.items(), key=lambda x: (color_weights.get(x[0], 0), x[1]), reverse=True)

    # Output construction
    output = [[0 for _ in range(4)] for _ in range(4)]
    for col in range(4):
        column_colors = [sections[row][col] for row in range(4)]
        primary_colors = [pair[0] for pair in column_colors if pair[0] != 0]
        secondary_colors = [pair[1] for pair in column_colors if pair[1] != 0]
        
        if primary_colors:
            main_color = max(set(primary_colors), key=primary_colors.count)
            alt_color = max(set(secondary_colors), key=secondary_colors.count) if secondary_colors else 0
            
            for row in range(4):
                if row == 0 or row == 3:
                    output[row][col] = main_color
                else:
                    output[row][col] = alt_color if alt_color != 0 else main_color

    # Pattern enhancement
    diagonal_color = global_ranking[0][0]
    for i in range(4):
        output[i][i] = diagonal_color

    # Ensure all prominent colors are represented
    prominent_colors = [color for color, _ in global_ranking[:4]]
    for color in prominent_colors:
        if all(color not in row for row in output):
            # Find a suitable position to introduce the color
            for r in range(4):
                if len(set(output[r])) > 2:  # If row has more than 2 colors, we can replace one
                    least_common = min(set(output[r]), key=lambda x: global_counts[x])
                    output[r][output[r].index(least_common)] = color
                    break

    # Final balance and symmetry check
    for col in range(4):
        col_colors = [output[r][col] for r in range(4)]
        if len(set(col_colors)) == 1:
            # Introduce variation in the column
            output[1][col] = prominent_colors[1] if prominent_colors[1] != col_colors[0] else prominent_colors[2]
        
        # Ensure symmetry in columns
        if output[0][col] != output[3][col]:
            output[3][col] = output[0][col]

    # Ensure black (0) is represented if it was prominent in the input
    if global_counts[0] > sum(global_counts.values()) / 5:  # If black was more than 20% of non-green cells
        if 0 not in [cell for row in output for cell in row]:
            output[2][2] = 0  # Place black in the center-right position

    return ColoredGrid(values=output)
