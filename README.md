# KNN from Scratch: Diabetes & Pickleball 🏓

This is the repository for our Group 1 K-Nearest Neighbors (KNN) machine learning project. We built the entire KNN algorithm from scratch using just Python, Pandas, and NumPy. No `scikit-learn` shortcuts here!

We tested our algorithm on two datasets: the clinical dataset our professor gave us, and a custom sports analytics dataset we built ourselves.

## 📂 The Datasets

### 1. The Diabetes Dataset (Class Assignment)
This is a clinical dataset with 768 patient records. We used it to predict if a patient has diabetes based on 8 biological features (Glucose, BMI, Age, Insulin, etc.).
* **What we did:** We cleaned up impossible data (like a blood pressure or BMI of 0) using median imputation. Then, we applied **Z-Score Standardization** so features with huge numbers (like Insulin reaching 800+) didn't mathematically overpower smaller features.

### 2. Our Custom Pickleball Dataset 
We made our own dataset based on Pickleball players! We tracked 100 players to predict their **Skill Level** (`Beginner`, `Intermediate`, or `Advanced`).
* **The Features:** We used `Age`, `Reaction_Time_ms`, and `Training_Hours_Per_Week`. 
* **What we did:** We used **Min-Max Normalization** to squish all the variables into a 0.0 to 1.0 scale so that the high reaction time numbers didn't outweigh the low training hour numbers.

## ✨ What's in the Code?
* **Pure KNN Math:** We hard-coded the Euclidean distance calculations and the majority voting system. 
* **Logistic Regression Comparison:** We also coded a Logistic Regression model from scratch (using Gradient Descent) just to see how it compared to our KNN model on the diabetes data.
* **Visualizations:** We used `matplotlib` to graph the "Bias-Variance Tradeoff" (Accuracy vs. K-value) and created a custom scatter plot for our Pickleball players showing distinct skill clusters based on training and reaction time.

## 🧠 What We Learned
* **Accuracy isn't everything (Diabetes):** For the Diabetes data, $K=3$, $K=5$, and $K=7$ all gave us the exact same overall accuracy (74.68%). But when we looked at the Confusion Matrix, $K=3$ had fewer False Negatives (24 instead of 25). In a medical setting, missing a sick patient is way worse than a false alarm, so $K=3$ was actually the safer clinical choice!
* **Finding the Sweet Spot (Pickleball):** On our Pickleball data, the model hit a peak accuracy of **85.00%** at $K=5$ and $K=7$. We chose $K=5$ as the optimal value because it requires fewer computations. 
* **Realistic Mistakes:** Our Pickleball model never once confused an `Advanced` player for a `Beginner`. The only errors it made were between bordering skill levels (like mistaking a high-level Beginner for an Intermediate), which proves our Euclidean distance math actually captured realistic athletic progression.
