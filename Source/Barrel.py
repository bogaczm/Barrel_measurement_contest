import cv2
import numpy as np
import copy


treshold_border = 20
treshold_high = np.array([110,110,110]) #np.array([100,100,100]) 
treshold_low = np.array([0,0,0])

min_size = 70
min_size_big = 4000
max_size = 2800
max_size_small = 800
close = 20
gaussian = 2

barrel_dir = 'C:\\Users\\moko\\OneDrive\\StartUP\\Barrel_measurement_contest\\Source\\Barrel1.jpg'
barrel_lid_real = 53.0 / 2

original = cv2.imread(barrel_dir)
original = cv2.resize(original, (509, 672), interpolation=cv2.INTER_NEAREST) 

output_oryginal = original.copy()

gaussian= cv2.GaussianBlur(original,(15,15),gaussian)

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
size_filter = np.reshape(size_filter, (672, 509, 1))

size_filter1 = cv2.convertScaleAbs(size_filter1)
size_filter1 = np.reshape(size_filter1, (672, 509, 1))

size_filter2 = cv2.convertScaleAbs(size_filter2)
size_filter2 = np.reshape(size_filter2, (672, 509, 1))

edges = cv2.Canny(size_filter,100,200, apertureSize = 7)
edges1 = cv2.Canny(size_filter1,100,200, apertureSize = 7)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.5, minDist=400)  #cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int") 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output_oryginal, (x, y), r, (0, 255, 0), 1)
        cv2.rectangle(output_oryginal, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

barrel_lid = {'x': circles[0, 0], 'y': circles[0, 1], 'r': circles[0, 2]}

circles = cv2.HoughCircles(edges1, cv2.HOUGH_GRADIENT, dp=4, minDist=40)  #cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int") 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output_oryginal, (x, y), r, (0, 0, 255), 1)
        cv2.rectangle(output_oryginal, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

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
im_with_keypoints = cv2.drawKeypoints(size_filter2, keypoints, np.array([]), (0,0,255))

# x = int(keypoints[0].pt[0])
# y = int(keypoints[0].pt[1])
# r = int(keypoints[0].size/2)

barrel_small_cap = {'x': x, 'y': y, 'r': r}

cv2.circle(output_oryginal, (x, y), r, (255, 0, 255), 1)
cv2.rectangle(output_oryginal, (x - 1, y - 1), (x + 1, y + 1), (0, 128, 255), -1)

cv2.imshow('1. oryginal', original)
#cv2.imwrite('oryginal.jpg', original)
cv2.imshow('2. gaussian', gaussian)
#cv2.imwrite('gaussian.jpg', gaussian)
cv2.imshow('3. color_filtered', color_filter)
#cv2.imwrite('color_filtered.jpg', color_filter)
cv2.imshow('4.1 size_filtered_big', size_filter)
#cv2.imwrite('size_filtered_big.jpg', size_filter)
cv2.imshow('4.2 size_filtered_medium', size_filter1)
#cv2.imwrite('size_filtered_medium.jpg', size_filter1)
cv2.imshow('4.3 size_filtered_small', size_filter2)
#cv2.imwrite('size_filtered_small.jpg', size_filter2)
cv2.imshow('5.1 edges_big', edges)
#cv2.imwrite('edges_big.jpg', edges)
cv2.imshow('5.2 edges_medium', edges1)
#cv2.imwrite('edges_medium.jpg', edges1)
cv2.imshow('5.3 blob_search', im_with_keypoints)
#cv2.imwrite('blob_search.jpg', im_with_keypoints)

# big_cap_dist = np.sqrt((barrel_big_cap['x']-barrel_lid['x'])**2 + (barrel_big_cap['y']-barrel_lid['y'])**2)
# small_cap_dist = np.sqrt((barrel_small_cap['x']-barrel_lid['x'])**2 + (barrel_small_cap['y']-barrel_lid['y'])**2)

# map_pix_mm = barrel_lid_real/barrel_lid['r']

# big_cap_dist = big_cap_dist * map_pix_mm
# small_cap_dist = small_cap_dist * map_pix_mm

# str_big_cap = 'BigC:'+'{:.1f}'.format(big_cap_dist)+'mm'
# str_small_cap = 'SmallC:'+'{:.1f}'.format(small_cap_dist)+'mm'
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(output_oryginal, str_big_cap, (10,30), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(output_oryginal, str_small_cap, (10,60), font, 1, (255, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('OUTPUT', output_oryginal)
#cv2.imwrite('output.jpg', output_oryginal)



cv2.waitKey(0) 
