import cv2
import numpy as np
import copy

barrel_dir = 'Barrel.jpg'
barrel_lid_real = 57.0 / 2

def competition_barrel_measuring(dir, real_r):
    treshold_high = np.array([110,110,110]) #np.array([100,100,100]) 
    treshold_low = np.array([0,0,0])

    min_size = 60
    min_size_big = 1000
    max_size = 700
    max_size_small = 80
    gaussian = 2

    original = cv2.imread(barrel_dir)
    
    gaussian= cv2.GaussianBlur(original,(5,5),gaussian)

    color_filter = cv2.inRange(gaussian, treshold_low, treshold_high)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(color_filter, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1 
    size_filter = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size_big:
            size_filter[output == i + 1] = 255

    sizes = stats[1:, -1]; nb_components = nb_components - 1 
    size_filter1 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] <= max_size and sizes[i] >= min_size:
            size_filter1[output == i + 1] = 255

    sizes = stats[1:, -1]; nb_components = nb_components - 1 
    size_filter2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] <= max_size_small and sizes[i] >= min_size:
            size_filter2[output == i + 1] = 255

    size_filter = cv2.convertScaleAbs(size_filter)
    size_filter = np.reshape(size_filter, (373, 280, 1))

    size_filter1 = cv2.convertScaleAbs(size_filter1)
    size_filter1 = np.reshape(size_filter1, (373, 280, 1))

    size_filter2 = cv2.convertScaleAbs(size_filter2)
    size_filter2 = np.reshape(size_filter2, (373, 280, 1))

    edges = cv2.Canny(size_filter,100,200, apertureSize = 7)
    edges1 = cv2.Canny(size_filter1,100,200, apertureSize = 7)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=40)  #cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int") 
        # loop over the (x, y) coordinates and radius of the circles

    barrel_lid = {'x': circles[0, 0], 'y': circles[0, 1], 'r': circles[0, 2]}

    circles = cv2.HoughCircles(edges1, cv2.HOUGH_GRADIENT, dp=4, minDist=40)  #cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int") 
        # loop over the (x, y) coordinates and radius of the circles

    barrel_big_cap = {'x': circles[0, 0], 'y': circles[0, 1], 'r': circles[0, 2]}

    # for smallest hole it might be better to find the blob center
    params = cv2.SimpleBlobDetector_Params()
    # all filters are disabled 
    params.filterByColor = False
    params.filterByArea = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # only the circurality
    params.filterByCircularity = True


    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(size_filter2)

    x = int(keypoints[0].pt[0])
    y = int(keypoints[0].pt[1])
    r = int(keypoints[0].size/2)

    barrel_small_cap = {'x': x, 'y': y, 'r': r}


    big_cap_dist = np.sqrt((barrel_big_cap['x']-barrel_lid['x'])**2 + (barrel_big_cap['y']-barrel_lid['y'])**2)
    small_cap_dist = np.sqrt((barrel_small_cap['x']-barrel_lid['x'])**2 + (barrel_small_cap['y']-barrel_lid['y'])**2)

    map_pix_mm = barrel_lid_real/barrel_lid['r']

    big_cap_dist = big_cap_dist * map_pix_mm
    small_cap_dist = small_cap_dist * map_pix_mm

    return big_cap_dist, small_cap_dist


big_cap_dist, small_cap_dist = competition_barrel_measuring(barrel_dir, barrel_lid_real)

print(big_cap_dist)
print(small_cap_dist)



