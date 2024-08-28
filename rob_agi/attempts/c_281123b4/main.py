from rob_agi.colored_grid import ColoredGrid
from collections import Counter
from typing import List, Tuple

def solve_281123b4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 19x4 input grid into a 4x4 output grid through a multi-step process:
    1. Input Analysis: Divides the input into four sections and analyzes color frequencies, distributions, and patterns.
    2. Color Importance Scoring: Assigns importance to colors based on frequency, distribution, and pattern involvement.
    3. Pattern Recognition: Identifies significant patterns in each section.
    4. Output Construction: Builds a 4x4 grid representing key features of each input section.
    5. Inter-quadrant Continuity: Ensures color continuity between adjacent quadrants.
    6. Global Color Balance: Adjusts the output to match the overall color distribution of the input.
    7. Refinement: Makes final adjustments to enhance representation and coherence.
    The final 4x4 grid captures the essence of each input section while maintaining overall balance.
    """
    rows, cols = input_grid.get_dimensions()
    if rows != 4 or cols != 19:
        raise ValueError("Input grid must be 19x4")

    # Input Analysis
    sections = [
        [row[0:4] for row in input_grid.values],
        [row[5:9] for row in input_grid.values],
        [row[10:14] for row in input_grid.values],
        [row[15:19] for row in input_grid.values]
    ]

    def analyze_section(section: List[List[int]]) -> Tuple[List[Tuple[int, float]], List[str]]:
        colors = [cell for row in section for cell in row]
        color_counts = Counter(colors)
        total_cells = 16
        color_importance = [(color, count / total_cells * (1.2 if color != 0 else 1)) for color, count in color_counts.items()]
        color_importance.sort(key=lambda x: x[1], reverse=True)

        patterns = []
        if section[0][0] == section[1][1] == section[2][2] == section[3][3] != 0:
            patterns.append("diagonal")
        if section[0][3] == section[1][2] == section[2][1] == section[3][0] != 0:
            patterns.append("anti-diagonal")
        if any(all(cell == row[0] != 0 for cell in row) for row in section):
            patterns.append("horizontal")
        if any(all(row[col] == section[0][col] != 0 for row in section) for col in range(4)):
            patterns.append("vertical")

        return color_importance, patterns

    section_analyses = [analyze_section(section) for section in sections]

    # Output Construction
    output = [[0 for _ in range(4)] for _ in range(4)]

    for i, (color_importance, patterns) in enumerate(section_analyses):
        row, col = divmod(i, 2)
        primary_color = color_importance[0][0] if color_importance else 0
        secondary_color = color_importance[1][0] if len(color_importance) > 1 else 0

        # Place primary color based on patterns
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

        # Place secondary color
        empty_cells = [(r, c) for r in range(row*2, row*2+2) for c in range(col*2, col*2+2) if output[r][c] == 0]
        if empty_cells and secondary_color:
            r, c = empty_cells[0]
            output[r][c] = secondary_color

    # Inter-quadrant Continuity and Global Color Balance
    all_colors = set(color for analysis in section_analyses for color, _ in analysis[0] if color != 0)
    for row in range(4):
        for col in range(4):
            if output[row][col] == 0:
                neighbors = [output[r][c] for r in range(max(0, row-1), min(4, row+2)) 
                             for c in range(max(0, col-1), min(4, col+2)) 
                             if (r, c) != (row, col) and output[r][c] != 0]
                if neighbors:
                    output[row][col] = max(set(neighbors), key=neighbors.count)
                else:
                    output[row][col] = max(all_colors, key=lambda c: sum(importance for analysis in section_analyses for color, importance in analysis[0] if color == c))

    # Refinement
    for _ in range(2):
        for row in range(4):
            for col in range(4):
                neighbors = [output[r][c] for r in range(max(0, row-1), min(4, row+2)) 
                             for c in range(max(0, col-1), min(4, col+2)) if (r, c) != (row, col)]
                if len(set(neighbors)) < 2:
                    other_colors = list(all_colors - set(neighbors))
                    if other_colors:
                        output[row][col] = max(other_colors, key=lambda c: sum(importance for analysis in section_analyses for color, importance in analysis[0] if color == c))

    return ColoredGrid(values=output)
