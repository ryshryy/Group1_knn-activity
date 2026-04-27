import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SETUP: Load the Dataset
# =========================================================
# Note: Ensure the path matches your local setup
df = pd.read_csv('LABACT-KNN/diabetes-k-nn.csv')

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================================================
# PART 2: DATA PREPROCESSING
# =========================================================
print("--- PART 2.4: BEFORE PREPROCESSING ---")
print(df.describe().round(2))

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns_with_zeros:
    df[col] = df[col].mask(df[col] == 0)
    col_median = df[col].median()
    df[col] = df[col].fillna(col_median)

features = df.drop('Outcome', axis=1)
target = df['Outcome']
df_scaled = pd.DataFrame()

for col in features.columns:
    mean_val = features[col].mean()
    std_val = features[col].std(ddof=0)
    df_scaled[col] = (features[col] - mean_val) / std_val

df_scaled['Outcome'] = target

print("\n--- PART 2.4: AFTER PREPROCESSING (Imputation & Standardization) ---")
print(df_scaled.describe().round(2))

# =========================================================
# PART 3: KNN IMPLEMENTATION
# =========================================================
np.random.seed(42) 
df_shuffled = df_scaled.sample(frac=1).reset_index(drop=True)

split_idx = int(0.8 * len(df_shuffled))
train_data = df_shuffled.iloc[:split_idx]
test_data = df_shuffled.iloc[split_idx:]

print("\n" + "="*50)
print(f"PART 3.1: DATA SPLIT -> {len(train_data)} Training rows, {len(test_data)} Testing rows")
print("="*50)

def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2)**2))

# --- MANUAL COMPUTATION SECTION ---
print("\nPART 3.4: MANUAL COMPUTATION FOR ONE TEST INSTANCE")
test_instance = test_data.iloc[0].drop('Outcome').values
true_label = test_data.iloc[0]['Outcome']

print("Step A: Selected Test Instance (Features Scaled)")
print(test_data.iloc[0].drop('Outcome').round(3).to_dict())
print(f"   Actual Class: {true_label}\n")

print("Step B: Calculating Euclidean Distance to 10 Training Samples")
sample_train = train_data.head(10)
manual_distances = []

for idx, row in sample_train.iterrows():
    train_features = row.drop('Outcome').values
    train_class = row['Outcome']
    dist = euclidean_distance(test_instance, train_features)
    manual_distances.append({'Train_Row': idx, 'Distance': dist, 'Class': train_class})
    print(f"   -> Distance to Train Row {idx}: {dist:.4f} | Class: {train_class}")

print("\nStep C: Identify nearest neighbors (using K=3)")
distances_sorted = sorted(manual_distances, key=lambda x: x['Distance'])
votes = []
for i in range(3):
    neighbor = distances_sorted[i]
    print(f"   Neighbor {i+1}: Train Row {neighbor['Train_Row']} (Distance: {neighbor['Distance']:.4f}) -> Vote: {neighbor['Class']}")
    votes.append(neighbor['Class'])

predicted_class = max(set(votes), key=votes.count)
print(f"\nStep D: Final Prediction")
print(f"   Majority Vote: Class {predicted_class}")
print(f"   Prediction matches Actual? {predicted_class == true_label}")


# =========================================================
# PART 4: MODEL EVALUATION & VISUALIZATION
# =========================================================
print("\n" + "="*50)
print("PART 4: KNN EVALUATION & VISUALIZATION")
print("="*50)

train_features_all = train_data.drop('Outcome', axis=1).values
train_labels = train_data['Outcome'].values
test_features_all = test_data.drop('Outcome', axis=1).values
test_labels = test_data['Outcome'].values

k_range = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []

for k in k_range:
    TP = 0; TN = 0; FP = 0; FN = 0
    for i in range(len(test_features_all)):
        distances = np.sqrt(np.sum((train_features_all - test_features_all[i])**2, axis=1))
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        pred = np.bincount(nearest_labels.astype(int)).argmax()
        actual = test_labels[i]
        
        # Populate Confusion Matrix
        if pred == 1 and actual == 1: TP += 1
        elif pred == 0 and actual == 0: TN += 1
        elif pred == 1 and actual == 0: FP += 1
        elif pred == 0 and actual == 1: FN += 1
    
    acc = ((TP + TN) / len(test_labels)) * 100
    accuracies.append(acc)
    
    # Print detailed Confusion Matrix specifically for K=3, 5, 7 as requested by assignment
    if k in [3, 5, 7]:
        print(f"\nResults for K = {k}")
        print("-" * 20)
        print(f"Accuracy: {acc:.2f}%")
        print("Confusion Matrix:")
        print(f"                 Predicted 0 | Predicted 1")
        print(f"Actual 0 (Neg) |     {TN:^7} |     {FP:^7}")
        print(f"Actual 1 (Pos) |     {FN:^7} |     {TP:^7}")
        print("-" * 40)

# Generate the plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='teal', linewidth=2)
plt.title('KNN Accuracy vs. K Value', fontsize=14)
plt.xlabel('Number of Neighbors (K)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# =========================================================
# PART 5: COMPARISON WITH LOGISTIC REGRESSION 
# =========================================================
print("\n" + "="*50)
print("PART 5: COMPARISON - LOGISTIC REGRESSION FROM SCRATCH")
print("="*50)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, iters=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    
    for _ in range(iters):
        model = np.dot(X, w) + b
        predictions = sigmoid(model)
        
        dw = (1 / m) * np.dot(X.T, (predictions - y))
        db = (1 / m) * np.sum(predictions - y)
        
        w -= lr * dw
        b -= lr * db
    return w, b

weights, bias = train_logistic_regression(train_features_all, train_labels)

lr_probs = sigmoid(np.dot(test_features_all, weights) + bias)
lr_preds = [1 if p >= 0.5 else 0 for p in lr_probs]
lr_accuracy = (np.sum(lr_preds == test_labels) / len(test_labels)) * 100

print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}%")

print("\n" + "-"*30)
print(f"{'Algorithm':<20} | {'Accuracy':<10}")
print("-" * 30)
print(f"{'KNN (Best K)':<20} | {max(accuracies):.2f}%")
print(f"{'Logistic Regression':<20} | {lr_accuracy:.2f}%")
print("-" * 30)

# =========================================================
# PART 6: FINAL COMPARISON GRAPH
# =========================================================
best_knn = max(accuracies)  # FIXED bug here
labels = ['KNN (Best K)', 'Logistic Regression']
vals = [best_knn, lr_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, vals, color=['teal', 'orange'], edgecolor='black', width=0.5)
plt.ylabel('Accuracy (%)')
plt.title('Final Algorithm Comparison')
plt.ylim(0, 100)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.2f}%", ha='center', fontweight='bold')
plt.show()