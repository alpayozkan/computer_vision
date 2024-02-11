import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    dh1, dw1 = desc1.shape
    dh2, dw2 = desc2.shape
    
    diff = desc1.reshape(dh1, 1, dw1) - desc2
    dist = np.sum(diff*diff, axis=2)

    return dist

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        mins_desc1 = np.argmin(distances, axis=1) # q1, 1
        matches = np.array([np.arange(distances.shape[0]), mins_desc1]) # 2, q1
        matches = matches.T # q1, 2
        
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        mins_desc1 = np.argmin(distances, axis=1) # q1, 1
        matches1 = np.array([np.arange(distances.shape[0]), mins_desc1]) # 2, q1
        matches1 = matches1.T # q1, 2
    
        mins_desc2 = np.argmin(distances, axis=0) # q2, 1
        matches2 = np.array([mins_desc2, np.arange(distances.shape[1])]) # 2, q1
        matches2 = matches2.T # q2, 2
    
        matches = np.array([w for w in matches1 for u in matches2 if np.array_equal(w,u)])
        
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        # You may use np.partition(distances,2,axis=1)[:,1] to find the second smallest value over a row => Correction !! => emailed
        
        mins_desc1 = np.argmin(distances, axis=1) # q1, 1
        matches = np.array([np.arange(distances.shape[0]), mins_desc1])
        matches = matches.T
        # match1st = np.partition(distances,2,axis=1)[:,0]
        match2nd = np.partition(distances,2,axis=1)[:,1]
        
        ratio = distances[np.arange(mins_desc1.shape[0]), mins_desc1] / match2nd
        
        matches = matches[ratio < ratio_thresh]
        
    else:
        raise NotImplementedError
    return matches

