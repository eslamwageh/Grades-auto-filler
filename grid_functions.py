from commonfunctions import *

def get_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.7)
    edges = cv2.Canny(blurred, 50, 80)

    show_images([image, gray, blurred, edges], ['img', 'gray', 'blurred', 'edges'])

    return edges

def getPerspective(img):
    imgContours = img.copy()
    imgCorners = img.copy()
    
    edges = get_edges(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


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
    show_images([closed_edges], ['closed_edges'])
    show_images([imgContours,imgCorners], ['imgContours','imgCorners'])
    show_images([warped_img], ["warped"])
    return warped_img

def getLines(full_paper):
    edges = get_edges(full_paper)

    edges = borderTheImage(edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(closed_edges, rho=0.001, theta=np.pi/180, threshold=100, minLineLength=400, maxLineGap=10)

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.append(line[0])
        elif abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.append(line[0])

    clustered_horizontal = cluster_lines(horizontal_lines, orientation='horizontal', threshold=10)
    clustered_vertical = cluster_lines(vertical_lines, orientation='vertical', threshold=10)
    clustered_lines = clustered_horizontal + clustered_vertical



    show_images([closed_edges], ["closed edges"])

    showLines(full_paper, horizontal_lines, vertical_lines, lines, 0)
    showLines(full_paper, clustered_horizontal, clustered_vertical, clustered_lines, 1)

    return clustered_horizontal, clustered_vertical

def showLines(full_paper, horizontal_lines, vertical_lines, lines, clustered = 0):
    horizontal = full_paper.copy()
    vertical = full_paper.copy()
    output = full_paper.copy()

    if horizontal_lines is not None:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(horizontal, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if vertical_lines is not None:
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(vertical, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            if (clustered == 0):
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
    if (clustered == 0):
        show_images([horizontal, vertical, output], ["horizontal", "vertical", "lines"])
    else:
        show_images([horizontal, vertical, output], ["clustered horizontal", "clustered vertical", "clustered lines"])

def borderTheImage(image, top = 3, bottom = 3, right = 5, left = 2):
    # Fill the borders with ones
    image[:top, :] = 255  # Top border
    image[-bottom:, :] = 255  # Bottom border
    image[:, :left] = 255  # Left border
    image[:, -right:] = 255  # Right border
    return image

def cluster_lines(lines, orientation='horizontal', threshold=10, shifting = 2):
    """
    Cluster lines that are close to each other.

    Parameters:
    - lines: list of lines [x1, y1, x2, y2]
    - orientation: 'horizontal' or 'vertical'
    - threshold: distance threshold for clustering

    Returns:
    - clustered_lines: list of clustered lines
    """
    if orientation == 'horizontal':
        # Use the y-coordinate for clustering
        lines_sorted = sorted(lines, key=lambda line: line[1])
    else:
        # Use the x-coordinate for clustering
        lines_sorted = sorted(lines, key=lambda line: line[0])
    
    clustered_lines = []
    cluster = []
    
    for line in lines_sorted:
        if not cluster:
            cluster.append(line)
            continue
        
        if orientation == 'horizontal':
            # Compare y-coordinates
            if abs(line[1] - cluster[-1][1]) < threshold:
                cluster.append(line)
            else:
                # Merge the current cluster
                avg_y = int(np.mean([l[1] for l in cluster])) - shifting
                x_start = min(l[0] for l in cluster)
                x_end = max(l[2] for l in cluster)
                clustered_lines.append([x_start, avg_y, x_end, avg_y])
                cluster = [line]
        else:
            # Compare x-coordinates
            if abs(line[0] - cluster[-1][0]) < threshold:
                cluster.append(line)
            else:
                # Merge the current cluster
                avg_x = int(np.mean([l[0] for l in cluster])) - shifting
                y_start = min(l[1] for l in cluster)
                y_end = max(l[3] for l in cluster)
                clustered_lines.append([avg_x, y_start, avg_x, y_end])
                cluster = [line]
    
    # Merge the last cluster
    if cluster:
        if orientation == 'horizontal':
            avg_y = int(np.mean([l[1] for l in cluster])) - shifting
            x_start = min(l[0] for l in cluster)
            x_end = max(l[2] for l in cluster)
            clustered_lines.append([x_start, avg_y, x_end, avg_y])
        else:
            avg_x = int(np.mean([l[0] for l in cluster])) - shifting
            y_start = min(l[1] for l in cluster)
            y_end = max(l[3] for l in cluster)
            clustered_lines.append([avg_x, y_start, avg_x, y_end])
    
    return clustered_lines

def extractCells(full_paper, clustered_horizontal, clustered_vertical):
    # Sort the horizontal and vertical lines
    clustered_horizontal = sorted(clustered_horizontal, key=lambda line: line[1])  # Sort by y
    clustered_vertical = sorted(clustered_vertical, key=lambda line: line[0])      # Sort by x

    # Number of rows and columns
    num_rows = len(clustered_horizontal) - 1
    num_cols = len(clustered_vertical) - 1

    print(f"Table has {num_rows} rows and {num_cols} columns.")

    cells = []

    for i in range(num_rows):
        row_cells = []
        for j in range(num_cols):
            # Top-left corner
            x1 = clustered_vertical[j][0]
            y1 = clustered_horizontal[i][1]
            
            # Bottom-right corner
            x2 = clustered_vertical[j + 1][0]
            y2 = clustered_horizontal[i + 1][1]
            
            # Crop the cell from the original image
            cell = full_paper[y1:y2, x1:x2]
            row_cells.append(cell)
            show_images([cell], [f"{i}, {j}"])
        cells.append(row_cells)
    
    print(f"Extracted {len(cells) * len(cells[0])} cells.")
    return cells
    



# def find_intersections(horizontal, vertical):
#     """
#     Find intersection points between horizontal and vertical lines.

#     Parameters:
#     - horizontal: list of horizontal lines [x1, y, x2, y]
#     - vertical: list of vertical lines [x, y1, x, y2]

#     Returns:
#     - intersections: list of (x, y) tuples
#     """
#     intersections = []
#     for h_line in horizontal:
#         _, y, _, _ = h_line
#         for v_line in vertical:
#             x, _, _, _ = v_line
#             intersections.append((x, y))
#     return intersections


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