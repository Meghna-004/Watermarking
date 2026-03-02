import cv2
import numpy as np


class ImageAligner:

    @staticmethod
    def align(original, attacked):

        gray1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(attacked, cv2.COLOR_BGR2GRAY)

        # ORB detector
        orb = cv2.ORB_create(nfeatures=6000)

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return attacked  # fallback

        # KNN matching instead of crossCheck
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test (important)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 15:
            return attacked  # not enough reliable matches

        pts1 = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        pts2 = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        if H is None:
            return attacked

        # Ensure enough inliers
        inliers = mask.ravel().tolist().count(1)
        if inliers < 10:
            return attacked

        height, width = original.shape[:2]
        aligned = cv2.warpPerspective(attacked, H, (width, height))

        return aligned