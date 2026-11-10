import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n_samples, n_features = X.shape 
    y_2d = y.reshape(-1,1)

    final_feature = -1.0
    final_threshold = -1.0
    final_gini = 1.0

    for feat_idx in range(n_features):
        feature = X[:, feat_idx: feat_idx+1]
        unique_val, counts = np.unique(feature, return_counts=True)
        unique_val = unique_val.reshape(1, -1)
        unique_val2d = np.tile(unique_val, (n_samples, 1))
        _, n_unique_val = unique_val.shape 
        feature = np.tile(feature, (1, n_unique_val))

        mask1 = feature > unique_val2d
        mask2 = feature <= unique_val2d
        
        # Count ones: sample must be in mask AND y must equal 1
        one_counts1 = np.sum(mask1 & (y_2d == 1), axis=0)
        zero_counts1 = np.sum(mask1 & (y_2d == 0), axis=0)
        
        total_count1 = zero_counts1 + one_counts1
        total_count1_num = total_count1.copy()
        total_count1[total_count1 == 0] = 1
        gini_scores1 = 1 - (zero_counts1/total_count1)**2 - (one_counts1/total_count1)**2

        one_counts2 = np.sum(mask2 & (y_2d == 1), axis=0)
        zero_counts2 = np.sum(mask2 & (y_2d == 0), axis=0)
        
        total_count2 = zero_counts2 + one_counts2
        total_count2_num = total_count2.copy()
        total_count2[total_count2 == 0] = 1
        gini_scores2 = 1 - (zero_counts2/total_count2)**2 - (one_counts2/total_count2)**2

        gini_scores = gini_scores1 *(total_count1_num/n_samples) + gini_scores2 * (total_count2_num/n_samples)

        min_idx = np.argmin(gini_scores)
        best_threshold = unique_val.flatten()[min_idx]
        gini = gini_scores.flatten()[min_idx]
        
        # Using strict '<' naturally preserves the first encountered split in case of ties
        if gini < final_gini:
            final_feature = feat_idx
            final_gini = gini 
            final_threshold = best_threshold

    return (final_feature, final_threshold)

       












