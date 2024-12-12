from commonfunctions import *

def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.7)
    edges = cv2.Canny(blurred, 50, 120)

    return edges

def getPerspective(img):
    imgContours = img.copy()
    imgCorners = img.copy()
    
    edges = get_edges(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    dilated_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    rectContours = rectContour(contours)

    cv2.drawContours(imgContours, rectContours, -1, (0, 255, 0), 2)

    thirdBiggestRect = getCornerPoints(rectContours[2])

    cv2.drawContours(imgCorners, thirdBiggestRect, -1, (0, 255, 0), 10)

    thirdBiggestRect = reorder(thirdBiggestRect)

    pt1 = np.float32(thirdBiggestRect)
    pt2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    # Apply the perspective transformation to the image
    warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))


    # SHOWING THE IMAGES FOR CLARITY 

    # show_images([img, gray, blurred], ['img', 'gray', 'blurred'])
    # show_images([edges, dilated_edges], ['edges', 'dilated_edges'])
    # show_images([imgContours,imgCorners], ['imgContours','imgCorners'])
    # show_images([warped_img], ["warped"])
    return warped_img

def getQuestions(full_paper):
    full_paper_contours = full_paper.copy()
    full_paper_largest_contour = full_paper.copy()

    edges = get_edges(full_paper)

    # Create a mask with zeros at the borders and ones elsewhere
    height,width = edges.shape
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[10:-10,10:-10] = 1  # Leave a 1-pixel border untouched

    # Dilate the edges
    print(edges.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertically_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
    show_images([full_paper,vertically_closed], ["full paper", "vertically_closed"])


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
    closed = cv2.morphologyEx(vertically_closed, cv2.MORPH_CLOSE, kernel, iterations=3)


    # # Apply the border mask to retain original borders
    closed_paper = np.where(border_mask, closed, edges)

    closed_paper[:10,:] = 0
    closed_paper[-10:, :] = 0
    closed_paper[:, :10] = 0
    closed_paper[:, -10:] = 0
    paper_contours, _ = cv2.findContours(closed_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # paper_rectContours = rectContour(paper_contours)
    cv2.drawContours(full_paper_contours, paper_contours, -1, (0, 255, 0), 3)
    # paper_largest_contour = paper_rectContours[0]
    paper_largest_contour = getLowerBiggestContour(paper_contours)
    cv2.drawContours(full_paper_largest_contour, [paper_largest_contour], -1, (0, 255, 0), 3)

    x, y, w, h = cv2.boundingRect(paper_largest_contour)
    questions = full_paper[y:y+h, x:x+w] 

    # SHOWING THE IMAGES FOR CLARITY 

    show_images([edges, closed_paper, full_paper_contours, full_paper_largest_contour, questions], ["edges","closed", "contours", "largest_contour", "questions"])
    return questions



# def getID(full_paper):
#     full_paper_contours = full_paper.copy()
#     full_paper_largest_contour = full_paper.copy()

#     gray = cv2.cvtColor(full_paper, cv2.COLOR_RGB2GRAY)
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0.7)
#     edges = cv2.Canny(blurred, 50, 120)


#     # Create a mask with zeros at the borders and ones elsewhere
#     height,width = edges.shape
#     border_mask = np.zeros((height, width), dtype=np.uint8)
#     border_mask[10:-10,10:-10] = 1  # Leave a 1-pixel border untouched

#     # Dilate the edges
#     print(edges.shape)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
#     vertically_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
#     show_images([vertically_closed], ["vertically_closed"])


#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 5))
#     closed = cv2.morphologyEx(vertically_closed, cv2.MORPH_CLOSE, kernel, iterations=3)



#     # # Apply the border mask to retain original borders
#     closed_paper = np.where(border_mask, closed, edges)

#     closed_paper[:10,:] = 0
#     closed_paper[-10:, :] = 0
#     closed_paper[:, :10] = 0
#     closed_paper[:, -10:] = 0
#     paper_contours, _ = cv2.findContours(closed_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # paper_rectContours = rectContour(paper_contours)
#     cv2.drawContours(full_paper_contours, paper_contours, -1, (0, 255, 0), 3)
#     # paper_largest_contour = paper_rectContours[0]
#     paper_largest_contour = getLowerBiggestContour(paper_contours)
#     cv2.drawContours(full_paper_largest_contour, [paper_largest_contour], -1, (0, 255, 0), 3)


#     x, y, w, h = cv2.boundingRect(paper_largest_contour)
#     questions = full_paper[y:y+h, x:x+w] 

#     # SHOWING THE IMAGES FOR CLARITY 

#     show_images([edges, closed_paper, full_paper_contours, full_paper_largest_contour, questions], ["edges","closed", "contours", "largest_contour", "questions"])
#     return questions