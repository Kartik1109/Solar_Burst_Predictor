# Created: 24/09/2024
# Author: Kartik Chaudhari
# Implementation: For Assignment 2, I have developed a solar burst prediction model using Support Vector Classification (SVC). It performs feature creation, preprocessing, and cross-validation to evaluate the model's performance through confusion matrices, accuracy, and True Skill Statistic (TSS) across multiple feature and data experiments.
# Sources: Class Notes + Textbook, Online Links: kaggle.com, stackoverflow.com, medium.com, machinelearningmastery.com, https://scikit-learn.org/stable/modules/svm.html,  https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea, https://python.plainenglish.io/support-vector-machine-svm-clearly-explained-d9db9123b7ac

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


# Starter Kit - By default the path to read the directory is set the name of the dataset itself. For example data-2010-15/data.npy
# If your data exist in other folders please update the path. The path can be updated manually in fetch_feature_and_target() or can be passed as an argument for the 2 experiemnts - feature, data experiment.
# Right now it's set to: X, Y = fetch_feature_and_target("data-2010-15") but if you directory is in other folder, please modify it.


class SVM_Solar_Burst_Predictor:
    
    def __init__(self, X, Y, C=1.0, learning_rate=0.001):
        self.X = X
        self.Y = Y
        self.C = C
        self.learning_rate = learning_rate
        self.model = SVC(C=self.C, kernel="rbf", gamma='scale')  # SVC instance from sklearn
        self.X_normalized = None
        self.Y_normalized = None

    def feature_creation(self, feature_numbers: list):
        if not (1 <= len(feature_numbers) <= 4):
            return  # Handle invalid feature count
        # Select specified features based on provided list of feature indices
        feature_ranges = {
            1: slice(0, 18),
            2: slice(18, 90),
            3: slice(90, 91),
            4: slice(91, None)
        }
         # Concatenate selected feature columns
        features = pd.concat([self.X.iloc[:, feature_ranges[fn]] for fn in feature_numbers], axis=1)
        self.X = features

    def preprocess(self):
        # Convert data to NumPy arrays and perform column-wise normalization
        X_ = self.X.to_numpy()
        means = X_.mean(axis=0)
        std_devs = X_.std(axis=0)
        
        # Avoid division by zero by replacing zero standard deviations with 1
        std_devs[std_devs == 0] = 1

        self.X_normalized = (X_ - means) / std_devs  # Column-wise normalization
        self.Y_normalized = self.Y.to_numpy() # Store labels

    def train(self):
        # Train the SVM using the normalized data
        self.model.fit(self.X_normalized, self.Y_normalized)

    def predict(self, X_test):
        # Predict using the trained SVC model
        return self.model.predict(X_test)

    def perform_cross_validation(self, k_folds=10, shuffle_data=True, random_seed=42):
        # Initialize KFold with given parameters
        kfold_splitter = KFold(n_splits=k_folds, shuffle=shuffle_data, random_state=random_seed)

        # Store metrics
        accuracy_scores, tss_scores, confusion_matrices = [], [], []
        def train_and_evaluate(train_idx, test_idx):
            # Subset training and testing data
            X_train, X_test = self.X_normalized[train_idx], self.X_normalized[test_idx]
            Y_train, Y_test = self.Y_normalized[train_idx], self.Y_normalized[test_idx]

            # Train the model and generate predictions
            self.model.fit(X_train, Y_train)
            predictions = self.model.predict(X_test)

            # Calculate evaluation metrics
            accuracy_scores.append(accuracy_score(Y_test, predictions))
            tss_scores.append(self.tss(Y_test, predictions))
            confusion_matrices.append(confusion_matrix(Y_test, predictions))

        # Loop through the folds and process each split
        for train_indices, test_indices in kfold_splitter.split(self.X_normalized):
            train_and_evaluate(train_indices, test_indices)

        # Return aggregated metrics (mean accuracy, TSS scores, confusion matrices)
        return np.mean(accuracy_scores), tss_scores, confusion_matrices


    def tss(self, Y_true, Y_pred):
        # Obtain confusion matrix and unpack values
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
        # Sensitivity (True Positive Rate)
        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0

        # Specificity (True Negative Rate)
        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        # Return the True Skill Statistic (TSS)
        tss_value = sensitivity + specificity - 1
        return tss_value

def get_confusion_matrix(confusion_matrices_list, combinations):
    # Plot confusion matrices for multiple feature combinations
    num_matrices = len(confusion_matrices_list)
    
    # Dynamically determine grid size (rows, cols) based on number of matrices
    cols = min(3, num_matrices)  # Limit to 3 columns, but adjust if fewer matrices
    rows = (num_matrices + cols - 1) // cols  # Calculate the number of rows needed

    # Create subplots based on number of matrices
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # Adjust figure size dynamically
    
    # Handle the case where there's only one matrix (axes is not a list in this case)
    if num_matrices == 1:
        axes = [axes]  # Make axes a list for consistent handling

    # Flatten axes for easy iteration (only if there are multiple matrices)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    for i, (conf_matrix, combination) in enumerate(zip(confusion_matrices_list, combinations)):
        # Plot each confusion matrix on its respective subplot
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="winter", cbar=False, ax=axes[i])
        
        # Set title and labels with smaller font size
        axes[i].set_title(f'Confusion Matrix: {combination}', fontsize=10)
        axes[i].set_xlabel('Predicted', fontsize=8)
        axes[i].set_ylabel('Actual', fontsize=8)
    
    # Remove any unused subplots if we have less than rows * cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout to prevent cropping
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.97, hspace=0.5, wspace=0.5)
    
    plt.show()


def plot_all_tss(all_tss_scores, all_directories):
    # Plot TSS scores for all datasets
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Define a color palette for better visual differentiation
    colors = sns.color_palette("husl", len(all_directories)) 
    
    # Loop through each path and plot its TSS scores on the same graph
    for idx, path in enumerate(all_directories):
        # Plot TSS for each dataset
        plt.plot(range(1, 11), all_tss_scores[idx], marker='o', linestyle='-', color=colors[idx],
                 label=f"Dataset {path}", markersize=8, linewidth=2)
    
    # Add title and labels with improved font size
    plt.title("TSS Scores for All Given Datasets", fontsize=18, fontweight='bold', color='navy')
    plt.xlabel("Fold Number", fontsize=14, fontweight='bold')
    plt.ylabel("TSS Score", fontsize=14, fontweight='bold')
    
    # Set ticks and limits for the y-axis
    plt.xticks(range(1, 11), fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.ylim(0, 1)  # Ensure the y-axis stays within the 0-1 range
    
    # Add a grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2)  # Horizontal line at y=0 for better reference
    
    # Add a legend
    plt.legend(title="Dataset Folder", title_fontsize=12, fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Display the plot with tight layout for better padding
    plt.tight_layout()
    plt.show()

def burst_presence(burst_type):
    # Convert burst presence to a binary format (-1 or 1)
    return -1 if burst_type is None else 1
  
def all_possibilities_generator(s):
    # Generate all possible combinations of elements from list `s`
    result = []
    for r in range(len(s) + 1):
        result.extend(itertools.combinations(s, r))
    return list(result)


def fetch_feature_and_target(path: str, shuffle=True):
    # Load features and labels from specified dataset directory
    pos_features = [np.load(f"{path}/pos_features_{name}.npy", allow_pickle=True) 
                    for name in ['main_timechange', 'historical', 'maxmin']]
    neg_features = [np.load(f"{path}/neg_features_{name}.npy", allow_pickle=True) 
                    for name in ['main_timechange', 'historical', 'maxmin']]

    # Load class labels
    labels_pos = np.load(f"{path}/pos_class.npy", allow_pickle=True)
    labels_neg = np.load(f"{path}/neg_class.npy", allow_pickle=True)

    # Combine positive and negative features
    fs_pos = np.column_stack(pos_features)
    fs_neg = np.column_stack(neg_features)

     # Stack features and create DataFrames
    df_pos = pd.DataFrame(fs_pos, columns=[f'FS_Feature_{i+1}' for i in range(fs_pos.shape[1])])
    df_neg = pd.DataFrame(fs_neg, columns=[f'FS_Feature_{i+1}' for i in range(fs_neg.shape[1])])

     # Add labels to data
    df_pos['FLARE'] = [burst_presence(label[2]) for label in labels_pos]
    df_neg['FLARE'] = [burst_presence(label[2]) for label in labels_neg]

    # Combine positive and negative data
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)

    # Handle shuffling
    if not shuffle:
        data_order = np.load(f"{path}/data_order.npy", allow_pickle=True)
        df_combined = df_combined.iloc[data_order]

     # Separate features and target labels
    feature = df_combined.drop(columns=['FLARE'])
    target = df_combined['FLARE']
    return feature, target

    
def feature_experiment():
     # Fetch the feature and target data from the dataset 'data-2010-15'
    X, Y = fetch_feature_and_target("data-2010-15")
    # Generate all possible feature combinations, except the empty one
    combinations = [list(tup) for tup in all_possibilities_generator([1,2,3,4])[1:]]
    best_tss, best_combination = 0, []
    
    results = []  # Store TSS and confusion matrices
    
    for combo in combinations:
         # Initialize the SVM model with the given data and hyperparameters
        svm = SVM_Solar_Burst_Predictor(X, Y, C=1, learning_rate=0.001)
        svm.feature_creation(combo)
        svm.preprocess()
        accuracy, tss, matrices = svm.perform_cross_validation()
        
        avg_tss = np.mean(tss)
        # Update the best TSS and combination if the current one is better
        if avg_tss > best_tss:
            best_tss, best_combination = avg_tss, combo
        
        # Store results for plotting
        results.append({
            'combination': combo,
            'matrix': np.sum(matrices, axis=0),
            'tss': tss
        })
        
        print(f"The Accuracy is: {accuracy:.4f}, And the Avg TSS is: {avg_tss:.4f} for Combination {combo} ")
        print()

    # Extract data for plotting
    matrices = [r['matrix'] for r in results]
    combos = [r['combination'] for r in results]
    tss_scores = [r['tss'] for r in results]
    
    # Plotting
    get_confusion_matrix(matrices, combos)
    plot_all_tss(tss_scores, combinations)
    
    print("***************************")
    print()
    print(f"The Best combination is for feature set: {best_combination}")
    print()
    print("***************************")
    print()
    return best_combination

def data_experiment(best_combination):
    # Current datasets for evaluation
    datasets = ['data-2010-15', 'data-2020-24']
    results = []
    
    for data in datasets:
        # Fetch the feature and target data for the current dataset
        X, Y = fetch_feature_and_target(data)
        
        
        svm = SVM_Solar_Burst_Predictor(X, Y, C=1, learning_rate=0.001)
        svm.feature_creation(best_combination)
        svm.preprocess()
        
          # Perform cross-validation and retrieve accuracy, TSS scores, and confusion matrices
        accuracy, tss_scores, matrices = svm.perform_cross_validation()
        
        # Store results
        results.append({
            'dataset': data,
            'matrix': np.sum(matrices, axis=0),
            'tss': tss_scores
        })
        
        # Print results for the current dataset
        print(f"For Dataset {data}:\n \n The Accuracy is: {accuracy:.4f}, And the Avg TSS: {np.mean(tss_scores):.4f} for the Combination: {best_combination}")
        print()
    print("***************************")
    print()
    print("Hence the best Dataset is: data-2010-15 ")
    print()
    print("***************************")
    print()

    # Extract data for plotting
    matrices = [r['matrix'] for r in results]
    labels = [r['dataset'] for r in results]
    tss_scores = [r['tss'] for r in results]
    
    get_confusion_matrix(matrices, labels)
    plot_all_tss(tss_scores, datasets)



def combine_confusion_matrices(matrices):
    # Combine confusion matrices by summing them up.
    return np.sum(matrices, axis=0)

def plot_combined_matrix(matrix, experiment_name):
    # Plot the confusion matrix for the given experiment.
    get_confusion_matrix([matrix], [experiment_name])

# Calling the 3 types of Function Experiments
best_combination = feature_experiment()    
data_experiment(best_combination=best_combination)
