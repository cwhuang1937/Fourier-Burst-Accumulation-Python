from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot, frame2tensor, read_image)


def register_burst_SuperGlue(burst, copy=False, force_cpu=False):
    if copy:
        burst = np.copy(burst)
    
    # Convert images to grayscale
    burst_gray = np.array([cv2.cvtColor(burst[i], cv2.COLOR_BGR2GRAY)
                          for i in range(burst.shape[0])])

    (height, width) = burst[0].shape[:2]

    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    config = {
        'superpoint': {
            'nms_radius': 4, #opt.nms_radius,
            'keypoint_threshold': 0.005, #opt.keypoint_threshold,
            'max_keypoints': -1 #opt.max_keypoints
        },
        'superglue': {
            'weights': 'indoor', #opt.superglue,
            'sinkhorn_iterations': 20, #opt.sinkhorn_iterations,
            'match_threshold': 0.2 #opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Warp the image pair.
    image0 = burst_gray[0]
    image0 = cv2.resize(image0, (image0.shape[1], image0.shape[0])).astype('float32')
    inp0 = frame2tensor(image0, device)
    for i in range(1, burst.shape[0]):
        print("process:"+str(i)+"/"+str(burst.shape[0]))
        image1 = burst_gray[i]
        image1 = cv2.resize(image1, (image1.shape[1], image1.shape[0])).astype('float32')
        inp1 = frame2tensor(image1, device)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                '../images/bookshelf/REG_001.jpg', '../images/bookshelf/REG_002.jpg'))
            exit(1)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # # Write the matches to disk.
        # out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
        #                 'matches': matches, 'match_confidence': conf}

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Find homography
        h, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)

        # Use homography
        burst[i] = cv2.warpPerspective(burst[i], h, (width, height))

        if False:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, './match.png', False,
                True, True, 'Matches', small_text)

    return burst

if __name__ == '__main__':
    img0 = cv2.imread('../images/bookshelf/REG_001.jpg')
    img1 = cv2.imread('../images/bookshelf/REG_002.jpg')
    img2 = cv2.imread('../images/bookshelf/REG_003.jpg')
    burst = np.array([img0, img1, img2])
    burst = register_burst_SuperGlue(burst, False)
    cv2.imshow('img0', burst[0])
    cv2.imshow('img1', burst[1])
    cv2.imshow('img2', burst[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()