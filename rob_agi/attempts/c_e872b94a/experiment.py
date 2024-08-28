from rob_agi.colored_grid import ColoredGrid

def analyze_example_3():
    input_grid = ColoredGrid(values=[
        [0, 5, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0],
        [0, 5, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0],
        [0, 5, 5, 0, 5, 5, 0, 5, 5, 0, 0, 0],
        [0, 0, 5, 0, 5, 0, 0, 5, 0, 0, 0, 0],
        [0, 0, 5, 0, 5, 0, 5, 5, 0, 0, 0, 0],
        [5, 5, 5, 0, 5, 0, 5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 5, 0, 0, 5, 5, 5],
        [0, 0, 0, 5, 5, 0, 5, 0, 0, 5, 0, 0],
        [0, 5, 5, 5, 0, 0, 5, 0, 0, 5, 0, 0]
    ])

    print("Distinct vertical regions per column:")
    for c in range(12):
        regions = 0
        prev_gray = False
        for r in range(9):
            if input_grid.values[r][c] == 5 and not prev_gray:
                regions += 1
            prev_gray = input_grid.values[r][c] == 5
        print(f"Column {c}: {regions}")

    print("\nDistinct connected regions:")
    visited = set()
    regions = 0

    def dfs(r, c):
        if (r, c) in visited or r < 0 or r >= 9 or c < 0 or c >= 12 or input_grid.values[r][c] != 5:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc)

    for r in range(9):
        for c in range(12):
            if input_grid.values[r][c] == 5 and (r, c) not in visited:
                regions += 1
                dfs(r, c)

    print(f"Total connected regions: {regions}")

if __name__ == "__main__":
    analyze_example_3()
