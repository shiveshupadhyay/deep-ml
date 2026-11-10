import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    
    points_arr = np.array(points)
    n, _ = points_arr.shape
    query_point_arr = np.array(query_point).reshape(-1, 2)
    query_point_arr = np.tile(query_point_arr, (n, 1))
    # print(points_arr.shape)
    # print(query_point_arr.shape, n)
    euclid_dist = np.sqrt(
        np.sum((points_arr - query_point_arr)**2, axis = 1)
        ).flatten()
    sorted_indices = np.argsort(euclid_dist, kind='stable')

    k_nearest_neigh = sorted_indices[:k]

    result = points_arr[k_nearest_neigh, :]

    return [tuple(i) for i in result.tolist()]
    







