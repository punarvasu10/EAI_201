import numpy as np

# Dataset
data = [
    {'area': 1200, 'rooms': 3, 'distance': 5, 'age': 10, 'price': 120},
    {'area': 1400, 'rooms': 4, 'distance': 3, 'age': 3, 'price': 150},
    {'area': 1600, 'rooms': 3, 'distance': 8, 'age': 20, 'price': 130},
    {'area': 1700, 'rooms': 5, 'distance': 2, 'age': 15, 'price': 180},
    {'area': 1850, 'rooms': 4, 'distance': 4, 'age': 7, 'price': 170}
]

# Calculate variance
def variance(prices):
    if len(prices) == 0: return 0
    mean = sum(prices) / len(prices)
    return sum((p - mean) ** 2 for p in prices) / len(prices)

# Root node
prices = [row['price'] for row in data]
root_variance = variance(prices)
print(f"Root Node - Mean Price: {sum(prices)/len(prices):.1f}, Variance: {root_variance:.1f}")

# Test splits for each feature
features = ['area', 'rooms', 'distance', 'age']
best_split = {'feature': None, 'value': None, 'variance': float('inf')}

for feature in features:
    values = sorted(list(set(row[feature] for row in data)))
    splits = [(values[i] + values[i+1])/2 for i in range(len(values)-1)]
    
    for split_val in splits:
        left_prices = [row['price'] for row in data if row[feature] <= split_val]
        right_prices = [row['price'] for row in data if row[feature] > split_val]
        
        w_variance = (len(left_prices)*variance(left_prices) + len(right_prices)*variance(right_prices)) / len(data)
        
        if w_variance < best_split['variance']:
            best_split = {'feature': feature, 'value': split_val, 'variance': w_variance}
        
        print(f"{feature} <= {split_val}: Weighted Variance = {w_variance:.2f}")

print(f"\nBest Split: {best_split['feature']} <= {best_split['value']}")
print(f"Variance Reduction: {root_variance - best_split['variance']:.1f}")

# Build first level
left_group = [row for row in data if row[best_split['feature']] <= best_split['value']]
right_group = [row for row in data if row[best_split['feature']] > best_split['value']]

print(f"\nLeft Group: {[row['price'] for row in left_group]}")
print(f"Right Group: {[row['price'] for row in right_group]}")