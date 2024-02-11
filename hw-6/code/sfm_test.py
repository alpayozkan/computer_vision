import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np


from impl.sfm.image import Image
from impl.sfm.io import ReadFeatureMatches, ReadKMatrix

from impl.util import MakeHomogeneous, HNormalize


from impl.sfm.corrs import Find2D3DCorrespondences, GetPairMatches, UpdateReconstructionState
from impl.sfm.vis import PlotImages, PlotWithKeypoints, PlotImagePairMatches, PlotCameras


# from impl.sfm.geometry import EstimateEssentialMatrix, DecomposeEssentialMatrix, TriangulatePoints

from impl.vis import Plot3DPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
    # TODO
    # Normalize coordinates (to points on the normalized image plane)
    # TODO CHECK: normalization -1,1 mi 0,1 mi ne kadar fark eder? l2 norm kurtarmaz mi
    
    # norm_mat1 = np.array([[2/im1.image.shape[1], 0, -1], [0, 2/im1.image.shape[0], -1], [0,0,1]])
    # norm_mat2 = np.array([[2/im2.image.shape[1], 0, -1], [0, 2/im2.image.shape[0], -1], [0,0,1]])

    normalized_kps1 = MakeHomogeneous(im1.kps, ax=1)
    normalized_kps2 = MakeHomogeneous(im2.kps, ax=1)

    # normalized_kps1 = normalized_kps1 @ norm_mat1.T
    # normalized_kps2 = normalized_kps2 @ norm_mat2.T

    # inv
    normalized_kps1 = normalized_kps1 @ np.linalg.inv(K).T
    normalized_kps2 = normalized_kps2 @ np.linalg.inv(K).T

    # Assemble constraint matrix as equation 2.1
    # TODO CHECK: buraya yp y order ini ters girersen gider => E.T calculate etmist olursun => Dikkat !!!
    constraint_matrix = np.zeros((matches.shape[0], 9))
    for i in range(matches.shape[0]): 
        # TODO
        # Add the constraints
        x1, x2 = matches[i]
        y1p, y2p, _ = normalized_kps1[x1]
        y1, y2, _ = normalized_kps2[x2]

        constraint_matrix[i][0] = y1p*y1
        constraint_matrix[i][1] = y1p*y2
        constraint_matrix[i][2] = y1p
        constraint_matrix[i][3] = y2p*y1
        constraint_matrix[i][4] = y2p*y2
        constraint_matrix[i][5] = y2p
        constraint_matrix[i][6] = y1
        constraint_matrix[i][7] = y2
        constraint_matrix[i][8] = 1

    # Solve for the nullspace of the constraint matrix
    _, _, vh = np.linalg.svd(constraint_matrix)
    vectorized_E_hat = vh[-1,:]

    # TODO
    # Reshape the vectorized matrix to it's proper shape again
    E_hat = vectorized_E_hat.reshape(3,3)

    # TODO
    # We need to fulfill the internal constraints of E
    # The first two singular values need to be equal, the third one zero.
    # Since E is up to scale, we can choose the two equal singluar values arbitrarilyu, s, v = np.linalg.svd(E_hat)
    u, s, v = np.linalg.svd(E_hat)
    s[0] = 1
    s[1] = 1
    s[2] = 0
    s = np.identity(3)*s
    E = u @ s @ v # np.matmul(np.matmul(u, s), v.T)

    # This is just a quick test that should tell you if your estimated matrix is not correct
    # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
    # You can adapt it to your assumptions.
    for i in range(matches.shape[0]):
        kp1 = normalized_kps1[matches[i,0],:]
        kp2 = normalized_kps2[matches[i,1],:]

        assert(abs(kp1.transpose() @ E @ kp2) < 0.01)
    
    return E

def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  points3D_homo = MakeHomogeneous(points3D, ax=1)
  proj_cam1 = points3D_homo @ P1.T
  proj_cam2 = points3D_homo @ P2.T

  mask_cam1 = proj_cam1[:,2]>0
  mask_cam2 = proj_cam2[:,2]>0

  # Filter points behind the first camera
  im1_corrs = im1_corrs[mask_cam1 & mask_cam2]
  im2_corrs = im2_corrs[mask_cam1 & mask_cam2]
  points3D = points3D[mask_cam1 & mask_cam2]

  # Filter points behind the second camera
  # im1_corrs = im1_corrs[mask_cam2]
  # im2_corrs = im2_corrs[mask_cam2]
  # points3D = points3D[mask_cam2]

  return points3D, im1_corrs, im2_corrs


def main():

    np.set_printoptions(linewidth=10000, edgeitems=100, precision=3)

    data_folder = '../data'
    image_names = [
        '0000.png',
        '0001.png',
        '0002.png',
        '0003.png',
        '0004.png',
        '0005.png',
        '0006.png',
        '0007.png',
        '0008.png',
        '0009.png']

    # Read images
    images = {}
    for im_name in image_names:
        images[im_name] = (Image(data_folder, im_name))

    # Read the matches
    matches = {}
    for image_pair in itertools.combinations(image_names, 2):
        matches[image_pair] = ReadFeatureMatches(image_pair, data_folder)

    K = ReadKMatrix(data_folder)

    init_images = [3, 4]

    # ------------------------------------------------------------------------------------
    # Visualize images and features
    # You can comment these lines once you verified that the images are loaded correctly

    # Show the images
    PlotImages(images)

    # Show the keypoints
    # for image_name in image_names:
    #     PlotWithKeypoints(images[image_name])

    # Show the feature matches
    # for image_pair in itertools.combinations(image_names, 2):
    #     PlotImagePairMatches(images[image_pair[0]], images[image_pair[1]], matches[(image_pair[0], image_pair[1])])
    #     gc.collect()
    # ------------------------------------------------------------------------------------
  
    e_im1_name = image_names[init_images[0]]
    e_im2_name = image_names[init_images[1]]
    e_im1 = images[e_im1_name]
    e_im2 = images[e_im2_name]
    e_matches = np.load('e_matches.npy')


    # TODO Estimate relative pose of first pair
    # Estimate Fundamental matrix
    E = EstimateEssentialMatrix(K, e_im1, e_im2, e_matches)

    # Extract the relative pose from the essential matrix.
    # This gives four possible solutions and we need to check which one is the correct one in the next step
    possible_relative_poses = DecomposeEssentialMatrix(E)

    # ------------------Finding the correct decomposition--------------------------------------
    # For each possible relative pose, try to triangulate points with function TriangulatePoints.
    # We can assume that the correct solution is the one that gives the most points in front of both cameras.
    max_points = 0
    best_pose = -1
    # Be careful not to set the transformation in the wrong direction
    # you can set the image poses in the images (image.SetPose(...))
    # Note that this pose is assumed to be the transformation from global space to image space
    # TODO
    for (r,t) in possible_relative_poses:
        r1, t1 = np.identity(r.shape[0]), np.zeros_like(t)
        e_im1.SetPose(r1, t1) # identity rotation, no translation -> assumed origin
        e_im2.SetPose(r, t)
        points3D, im1_corrs, im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)

        if len(points3D) > max_points:
            max_points = len(points3D)
            best_pose = (r,t)

    # TODO
    # Set the image poses in the images (image.SetPose(...))
    # Note that the pose is assumed to be the transformation from global space to image space
    r,t = best_pose
    r1, t1 = np.identity(r.shape[0]), np.zeros_like(t)        
    e_im1.SetPose(r1, t1) # identity
    e_im2.SetPose(r, t) # r,t on the second


    # Triangulate initial points
    points3D, im1_corrs, im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)

    # Add the new 2D-3D correspondences to the images
    e_im1.Add3DCorrs(im1_corrs, list(range(points3D.shape[0])))
    e_im2.Add3DCorrs(im2_corrs, list(range(points3D.shape[0])))

    # Keep track of all registered images
    registered_images = [e_im1_name, e_im2_name]

    for reg_im in registered_images:
        print(f'Image {reg_im} sees {images[reg_im].NumObserved()} 3D points')

    # Visualize
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    Plot3DPoints(points3D, ax3d)
    PlotCameras(images, registered_images, ax3d)

    # Delay termination of the program until the figures are closed
    # Otherwise all figure windows will be killed with the program
    plt.show(block=True)

if __name__ == '__main__':
  main()
