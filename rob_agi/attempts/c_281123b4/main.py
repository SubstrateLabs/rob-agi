from rob_agi.colored_grid import ColoredGrid
from collections import Counter
import random

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Input Preprocessing: Verifies input dimensions and divides into sections.
    2. Section Analysis: Analyzes color frequencies, distributions, and patterns in each section.
    3. Output Grid Construction: Builds a 4x4 grid based on section analysis.
    4. Color Placement: Places primary, secondary, and tertiary colors for each quadrant.
    5. Pattern Representation: Attempts to represent significant patterns from input sections.
    6. Inter-section Continuity: Ensures some color continuity between adjacent sections.
    7. Balance and Refinement: Adjusts for color variety and representation.
    8. Final Validation: Ensures output reflects input essence while maintaining coherence.
    The final 4x4 grid captures the key characteristics and patterns of each input section.
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
    section_patterns = []
    for section in sections:
        colors = [cell for row in section for cell in row if cell != 0]
        color_counts = Counter(colors)
        section_colors.append(sorted(color_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Simple pattern detection (e.g., diagonal, vertical, horizontal)
        patterns = []
        if section[0][0] == section[1][1] == section[2][2] == section[3][3]:
            patterns.append("diagonal")
        if section[0][3] == section[1][2] == section[2][1] == section[3][0]:
            patterns.append("anti-diagonal")
        if any(all(cell == row[0] for cell in row) for row in section):
            patterns.append("horizontal")
        if any(all(row[col] == section[0][col] for row in section) for col in range(4)):
            patterns.append("vertical")
        section_patterns.append(patterns)

    # Output Grid Construction
    output = [[0 for _ in range(4)] for _ in range(4)]

    # Color Placement and Pattern Representation
    for i, (colors, patterns) in enumerate(zip(section_colors, section_patterns)):
        row, col = divmod(i, 2)
        primary_color = colors[0][0] if colors else 0
        secondary_color = colors[1][0] if len(colors) > 1 else 0
        tertiary_color = colors[2][0] if len(colors) > 2 else 0

        # Place colors based on patterns
        if "diagonal" in patterns:
            output[row*2][col*2] = output[row*2+1][col*2+1] = primary_color
        elif "anti-diagonal" in patterns:
            output[row*2][col*2+1] = output[row*2+1][col*2] = primary_color
        elif "horizontal" in patterns:
            output[row*2][col*2] = output[row*2][col*2+1] = primary_color
        elif "vertical" in patterns:
            output[row*2][col*2] = output[row*2+1][col*2] = primary_color
        else:
            output[row*2][col*2] = primary_color

        # Place secondary and tertiary colors
        empty_cells = [(r, c) for r in range(row*2, row*2+2) for c in range(col*2, col*2+2) if output[r][c] == 0]
        if empty_cells and secondary_color:
            r, c = random.choice(empty_cells)
            output[r][c] = secondary_color
            empty_cells.remove((r, c))
        if empty_cells and tertiary_color:
            r, c = random.choice(empty_cells)
            output[r][c] = tertiary_color

    # Inter-section Continuity
    for row in range(4):
        for col in range(4):
            if row % 2 == 1 and col % 2 == 1:
                neighbors = [output[r][c] for r in [row-1, row+1] for c in [col-1, col+1] if 0 <= r < 4 and 0 <= c < 4]
                if neighbors:
                    output[row][col] = random.choice(neighbors)

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
