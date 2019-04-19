from scipy.spatial import distance as dist
import numpy as np
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def x_cord_contour(contours):
    ## returns the x coordinate for the contour centroid
    if cv2.contourArea(contours)>10:
        M = cv2.moments(contours)
        return(int(M['m10']/M['m00']))

# construct the argument parse and parse the arguments

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("images/Test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("test1",gray)
gray = cv2.GaussianBlur(gray, (7, 7), 0)


# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=5)
edged = cv2.erode(edged, None, iterations=3)

# find contours in the edge map
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

#sorted_ctrs = sorted(contours, key = cv2.contourArea, reverse = True)
sorted_ctrs = sorted(contours, key = x_cord_contour, reverse = False)
pixelsPerMetric = None

# loop over the contours individually
for c in sorted_ctrs:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	#box = cv2.cv.BoxPoints(box) 
	box = cv2.boxPoints(box) 
	box = np.array(box, dtype="int")

	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / 2.1

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	cv2.putText(orig, "Width : {:.1f}in".format(dimA),
		(15,30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "Height: {:.1f}in".format(dimB),
		(15,60), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
             
    # show the output image
	cv2.imshow("Image", orig)
	cv2.waitKey(0)
    
cv2.destroyAllWindows()