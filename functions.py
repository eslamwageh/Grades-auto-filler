from commonfunctions import *


def getPerspective(img):
    imgContours = img.copy()
    imgCorners = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0.7)

    edges = cv2.Canny(blurred, 50, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    rectContours = rectContour(contours)

    cv2.drawContours(imgContours, rectContours, -1, (0, 255, 0), 1)

    thirdBiggestRect = getCornerPoints(rectContours[2])

    cv2.drawContours(imgCorners, thirdBiggestRect, -1, (0, 255, 0), 10)

    thirdBiggestRect = reorder(thirdBiggestRect)

    pt1 = np.float32(thirdBiggestRect)
    pt2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    # Apply the perspective transformation to the image
    warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))


    show_images([img, gray, blurred], ['img', 'gray', 'blurred'])
    show_images([edges, dilated_edges], ['edges', 'dilated_edges'])
    show_images([imgContours,imgCorners], ['imgContours','imgCorners'])
    show_images([warped_img], ["warped"])
    return warped_img