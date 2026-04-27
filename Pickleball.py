import pandas as pd
import numpy as np

# ==========================================
# --- PART 2: DATA PREPROCESSING ---
# ==========================================
df = pd.read_csv('pickleball_knn_dataset.csv')

print("--- BEFORE PREPROCESSING (Raw Data) ---")
print(df.head())

# Feature Scaling (Min-Max Normalization)
features_to_scale = ['Age', 'Reaction_Time_ms', 'Training_Hours_Per_Week']
df_scaled = df.copy()

for col in features_to_scale:
    min_val = df_scaled[col].min()
    max_val = df_scaled[col].max()
    df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)

print("\n--- AFTER PREPROCESSING (Scaled Data) ---")
print(df_scaled.head())

# ==========================================
# --- PART 3: KNN IMPLEMENTATION ---
# ==========================================

# 1. SPLIT THE DATASET (80% Training, 20% Testing)
np.random.seed(42)  # Ensure we get the same random split every time
shuffled_indices = np.random.permutation(len(df_scaled))
split_idx = int(0.8 * len(df_scaled))  # 80% of 100 rows is 80

train_indices = shuffled_indices[:split_idx]
test_indices = shuffled_indices[split_idx:]

train_data = df_scaled.iloc[train_indices].reset_index(drop=True)
test_data = df_scaled.iloc[test_indices].reset_index(drop=True)

print(f"\n--- 1. DATASET SPLIT ---")
print(f"Training Samples: {len(train_data)} rows (80%)")
print(f"Testing Samples: {len(test_data)} rows (20%)")

# 4. FOR ONE SELECTED TEST INSTANCE
test_instance = test_data.iloc[0]  # Grab the very first person in the test set
print(f"\n--- 4. TEST INSTANCE (Mystery Player) ---")
print(
    f"Stats: Age={test_instance['Age']:.3f}, Reaction={test_instance['Reaction_Time_ms']:.3f}, Training={test_instance['Training_Hours_Per_Week']:.3f}")
print(f"Actual Skill Level: {test_instance['Skill_Level']}")

# Grab 10 training samples to compare against
sample_train = train_data.head(10)
distances = []

print("\nStep-by-Step Distance Computation to 10 Training Samples:")
# 3. USE EUCLIDEAN DISTANCE
for i, train_row in sample_train.iterrows():
    # Math: (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2
    age_diff_sq = (train_row['Age'] - test_instance['Age']) ** 2
    react_diff_sq = (train_row['Reaction_Time_ms'] - test_instance['Reaction_Time_ms']) ** 2
    train_diff_sq = (train_row['Training_Hours_Per_Week'] - test_instance['Training_Hours_Per_Week']) ** 2

    # Square Root of the sum
    dist = np.sqrt(age_diff_sq + react_diff_sq + train_diff_sq)
    distances.append((i, dist, train_row['Skill_Level']))

    print(f"Distance to Train Sample {i} ({train_row['Skill_Level']}): {dist:.4f}")

# Sort distances from shortest to longest
distances.sort(key=lambda x: x[1])

# 2. TRAIN KNN MODEL USING DIFFERENT K VALUES
print("\n--- IDENTIFYING NEAREST NEIGHBORS AND PREDICTED CLASS ---")
for k in [3, 5, 7]:
    print(f"\nEvaluating for K = {k}:")
    nearest_neighbors = distances[:k]

    votes = {}
    for neighbor in nearest_neighbors:
        idx, d, label = neighbor
        print(f"  Neighbor {idx}: Dist={d:.4f} -> Vote: {label}")
        votes[label] = votes.get(label, 0) + 1

    predicted_class = max(votes, key=votes.get)
    print(f"  => Majority Prediction: {predicted_class}")

    # ==========================================
    # --- PART 4: MODEL EVALUATION ---
    # ==========================================
    print("\n==========================================")
    print("--- PART 4: MODEL EVALUATION ---")
    print("==========================================")


    # Helper function for KNN
    def predict_knn(train_df, test_row, k_val):
        distances = []
        for i, train_row in train_df.iterrows():
            dist = np.sqrt((train_row['Age'] - test_row['Age']) ** 2 +
                           (train_row['Reaction_Time_ms'] - test_row['Reaction_Time_ms']) ** 2 +
                           (train_row['Training_Hours_Per_Week'] - test_row['Training_Hours_Per_Week']) ** 2)
            distances.append((dist, train_row['Skill_Level']))
        distances.sort(key=lambda x: x[0])

        top_k_labels = [label for dist, label in distances[:k_val]]
        votes = {}
        for label in top_k_labels:
            votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get)


    # Setup evaluation
    k_values_to_test = [3, 5, 7]
    all_predictions = {3: [], 5: [], 7: []}
    actual_labels = test_data['Skill_Level'].tolist()
    accuracies = {}

    # Evaluate all 20 Test Samples
    print("Calculating Accuracies on 20 Test Samples...")
    for k in k_values_to_test:
        correct = 0
        for idx, test_row in test_data.iterrows():
            pred = predict_knn(train_data, test_row, k)
            all_predictions[k].append(pred)
            if pred == test_row['Skill_Level']:
                correct += 1

        # Calculate and store accuracy
        accuracy = (correct / len(test_data)) * 100
        accuracies[k] = accuracy
        print(f"Accuracy for K={k}: {accuracy:.2f}%")

    # Identify the Best K
    best_k = max(accuracies, key=accuracies.get)
    print(f"\nBest performing K value: K={best_k} ({accuracies[best_k]:.2f}% accuracy)")

    # Print the Confusion Matrices for all K values
    unique_labels = ['Beginner', 'Intermediate', 'Advanced']
    header_str = "Actual \\ Predicted"

    for k in k_values_to_test:
        print(f"\n--- CONFUSION MATRIX (for K={k}) ---")

        conf_matrix = {actual: {pred: 0 for pred in unique_labels} for actual in unique_labels}
        for actual, pred in zip(actual_labels, all_predictions[k]):
            conf_matrix[actual][pred] += 1

        print(f"{header_str:<20} | {'Beginner':<12} | {'Intermediate':<12} | {'Advanced':<12}")
        print("-" * 65)
        for actual in unique_labels:
            row_str = f"{actual:<20} | "
            for pred in unique_labels:
                row_str += f"{conf_matrix[actual][pred]:<12} | "
            print(row_str)

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ==========================================
# --- BONUS: VISUALIZATIONS & COMPARISON ---
# ==========================================
print("\n==========================================")
print("--- BONUS: GRAPHS & MODEL COMPARISON ---")
print("==========================================")

# 1. Scatter Plot (Raw Data Visualization)
# We use the raw 'df' here (not the scaled one) so the axes make sense to humans
plt.figure(figsize=(8, 6))
colors = {'Beginner': 'red', 'Intermediate': 'blue', 'Advanced': 'green'}

for skill_level, color in colors.items():
    subset = df[df['Skill_Level'] == skill_level]
    plt.scatter(subset['Training_Hours_Per_Week'], subset['Reaction_Time_ms'],
                label=skill_level, color=color, alpha=0.7, edgecolors='k', s=80)

plt.title('Pickleball Skill Levels: Training vs. Reaction Time')
plt.xlabel('Training Hours Per Week')
plt.ylabel('Reaction Time (ms)')
plt.legend(title="Skill Level")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('scatter_plot.png') # Saves the image to your folder
print("Saved scatter plot to 'scatter_plot.png'")


# 2. Accuracy vs K Plot
plt.figure(figsize=(8, 6))
k_list = list(accuracies.keys())
acc_list = list(accuracies.values())

plt.plot(k_list, acc_list, marker='o', linestyle='-', color='purple', markersize=8, linewidth=2)
plt.title('KNN Model Accuracy vs. K Value')
plt.xlabel('K Value (Number of Neighbors)')
plt.ylabel('Accuracy (%)')
plt.xticks(k_list)
plt.ylim(0, 100) # Lock the Y-axis between 0 and 100%
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('accuracy_vs_k.png') # Saves the image to your folder
print("Saved Accuracy vs K plot to 'accuracy_vs_k.png'")


# 3. Algorithm Comparison (KNN vs Logistic Regression)
# We use the scaled data to train the Logistic Regression model for a fair fight
X_train = train_data[['Age', 'Reaction_Time_ms', 'Training_Hours_Per_Week']]
y_train = train_data['Skill_Level']
X_test = test_data[['Age', 'Reaction_Time_ms', 'Training_Hours_Per_Week']]
y_test = test_data['Skill_Level']

# Initialize and train Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict and calculate accuracy
lr_predictions = log_reg.predict(X_test)
lr_correct = sum(lr_predictions == y_test)
lr_accuracy = (lr_correct / len(y_test)) * 100

print(f"\n--- ALGORITHM COMPARISON RESULTS ---")
print(f"KNN Accuracy (Best K={best_k}): {accuracies[best_k]:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}%")