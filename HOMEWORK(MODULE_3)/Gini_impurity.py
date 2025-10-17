# Dataset
data = [
    {'hours': 2, 'pass': 0},
    {'hours': 4, 'pass': 0},
    {'hours': 6, 'pass': 1},
    {'hours': 8, 'pass': 1},
    {'hours': 10, 'pass': 1},
]

# Step 0: Print dataset
print("Dataset:")
for i, row in enumerate(data):
    print(f"Student {i+1}: {row['hours']} hours → {'Pass' if row['pass'] else 'Fail'}")

# Gini impurity function
def gini(groups):
    total = sum(len(group) for group in groups)
    gini_value = 0.0
    for group in groups:
        size = len(group)
        if size == 0: continue
        p1 = sum(row['pass'] for row in group) / size
        gini_value += (size / total) * (1 - (p1 ** 2 + (1-p1) ** 2))
    return gini_value

# Step 1: Find possible split points
splits = [3, 5, 7, 9]
print("\nStep 1: Possible splits:", splits)

# Step 2: Calculate Gini for each split
best_split = None
lowest_gini = 1.0

print("\nStep 2: Calculate Gini impurity for each split")
for s in splits:
    left = [row for row in data if row['hours'] <= s]
    right = [row for row in data if row['hours'] > s]
    weighted_gini = gini([left, right])
    print(f"Split at {s}: Left={[row['pass'] for row in left]}, Right={[row['pass'] for row in right]}, Weighted Gini={weighted_gini:.3f}")

    if weighted_gini < lowest_gini:
        lowest_gini = weighted_gini
        best_split = s

print("\nBest Split is at:", best_split, "with Gini:", round(lowest_gini, 3))

# Step 3: Decision Tree
print("\nDecision Tree:")
print(f"If Study Hours ≤ {best_split} → Predict: Fail (0)")
print(f"If Study Hours > {best_split} → Predict: Pass (1)")

# Step 4: Predictions
print("\nPredictions:")
test_hours = [1, 3, 5, 7, 9, 11]
for h in test_hours:
    prediction = "Fail" if h <= best_split else "Pass"
    print(f"Study {h} hours → {prediction}")