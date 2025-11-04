# signature_auth.py

import cv2
import numpy as np

# Initialize the ORB detector once
orb = cv2.ORB_create(nfeatures=500)

def extract_signature_features(image):
    """
    Processes a signature image to find keypoints and descriptors using ORB.
    This method is robust to translation, scale, and rotation.
    """
    if image is None:
        return None, None

    # 1. Standard Preprocessing for RGBA canvas images
    image_uint8 = np.array(image).astype(np.uint8)
    if image_uint8.shape[2] == 4:
        alpha = image_uint8[:, :, 3]
        mask = alpha > 10
        rgb_image = np.full((image_uint8.shape[0], image_uint8.shape[1], 3), 255, dtype=np.uint8)
        for c in range(3):
            rgb_image[:, :, c][mask] = image_uint8[:, :, c][mask]
    else:
        rgb_image = image_uint8

    # 2. Convert to grayscale, which is required for the ORB detector
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # 3. Find keypoints and compute their descriptors
    # The mask ensures we only find keypoints on the signature itself, not the blank background
    mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)[1]
    keypoints, descriptors = orb.detectAndCompute(gray_image, mask=mask)
    
    return keypoints, descriptors


def compare_signature_features(features1, features2):
    """
    Compares two sets of signature features (keypoints and descriptors).
    Returns a similarity score based on the ratio of good matches.
    """
    keypoints1, descriptors1 = features1
    keypoints2, descriptors2 = features2

    # Check for invalid inputs
    if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
        return 0.0

    # 1. Use a Brute-Force Matcher to find potential matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 2. Apply Lowe's ratio test to filter out bad matches
    good_matches = []
    
    # +++ THIS IS THE FIX +++
    # Add a safety check to ensure each match has two points (m and n) before unpacking.
    # This prevents the ValueError.
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    # 3. Calculate the final similarity score
    num_keypoints = min(len(keypoints1), len(keypoints2))
    if num_keypoints == 0:
        return 0.0

    similarity_score = len(good_matches) / num_keypoints
    
    return min(similarity_score, 1.0)