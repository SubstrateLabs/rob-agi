from rob_agi.colored_grid import ColoredGrid
from collections import Counter
import random

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Input Preprocessing: Verifies input dimensions and divides into sections.
    2. Section Analysis: Analyzes color frequencies and distributions in each section.
    3. Output Grid Construction: Builds a 4x4 grid based on section analysis.
    4. Color Placement: Places primary and secondary colors for each quadrant.
    5. Inter-section Influence: Considers colors from adjacent sections.
    6. Balance and Refinement: Adjusts for color variety and representation.
    7. Final Validation: Ensures output reflects input essence while maintaining coherence.
    The final 4x4 grid captures the key characteristics of each input section.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    # Input Preprocessing
    sections = [
        [row[0:4] for row in input_grid.values],
        [row[5:9] for row in input_grid.values],
        [row[10:14] for row in input_grid.values],
        [row[15:19] for row in input_grid.values]
    ]

    # Section Analysis
    section_colors = []
    for section in sections:
        colors = [cell for row in section for cell in row if cell != 0]
        color_counts = Counter(colors)
        section_colors.append(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))

    # Output Grid Construction
    output = [[0 for _ in range(4)] for _ in range(4)]

    # Color Placement and Inter-section Influence
    for i, colors in enumerate(section_colors):
        row, col = divmod(i, 2)
        primary_color = colors[0][0] if colors else 0
        secondary_color = colors[1][0] if len(colors) > 1 else 0

        # Place primary color
        output[row*2][col*2] = primary_color
        if random.random() < 0.7:  # 70% chance to place primary color twice
            output[row*2+1][col*2+1] = primary_color

        # Place secondary color
        if secondary_color:
            output[row*2+(1-row)][col*2+(1-col)] = secondary_color

        # Consider adjacent sections
        adjacent_sections = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for adj_row, adj_col in adjacent_sections:
            if 0 <= adj_row < 2 and 0 <= adj_col < 2:
                adj_colors = section_colors[adj_row*2 + adj_col]
                if adj_colors:
                    adj_color = adj_colors[0][0]
                    if adj_color not in [primary_color, secondary_color]:
                        empty_cells = [(r, c) for r in range(row*2, row*2+2) for c in range(col*2, col*2+2) if output[r][c] == 0]
                        if empty_cells:
                            r, c = random.choice(empty_cells)
                            output[r][c] = adj_color

    # Fill remaining spaces and balance
    all_colors = set(color for section in section_colors for color, _ in section)
    for row in range(4):
        for col in range(4):
            if output[row][col] == 0:
                output[row][col] = random.choice(list(all_colors))

    # Final Validation and Refinement
    for _ in range(2):
        for row in range(4):
            for col in range(4):
                neighbors = [output[r][c] for r in range(max(0, row-1), min(4, row+2)) 
                             for c in range(max(0, col-1), min(4, col+2)) if (r, c) != (row, col)]
                if len(set(neighbors)) < 2:
                    other_colors = list(all_colors - set(neighbors))
                    if other_colors:
                        output[row][col] = random.choice(other_colors)

    return ColoredGrid(values=output)
