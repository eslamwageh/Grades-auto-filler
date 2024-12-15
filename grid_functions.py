from commonfunctions import *
from joblib import dump, load
import pytesseract

def get_edges(image):
    blurred = cv2.GaussianBlur(image, (21, 21), 0.7)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    
    edges = cv2.Canny(gray,50, 80)

    show_images([image, gray, blurred, edges], ['img', 'gray', 'blurred', 'edges'])

    return edges

def getPerspective(img):
    imgContours = img.copy()
    imgCorners = img.copy()
    
    
    edges = get_edges(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    rectContours = rectContour(contours)

    cv2.drawContours(imgContours, rectContours, -1, (0, 255, 0), 2)

    thirdBiggestRect = getCornerPoints(rectContours[2])

    cv2.drawContours(imgCorners, thirdBiggestRect, -1, (0, 255, 0), 50)

    thirdBiggestRect = reorder(thirdBiggestRect)

    pt1 = np.float32(thirdBiggestRect)
    pt2 = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    # Apply the perspective transformation to the image
    warped_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))


    # SHOWING THE IMAGES FOR CLARITY 
    # show_images([closed_edges], ['closed_edges'])
    # show_images([imgContours,imgCorners], ['imgContours','imgCorners'])
    # show_images([warped_img], ["warped"])
    return warped_img

def getLines(full_paper):
    edges = get_edges(full_paper)

    edges = borderTheImage(edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=6)

    # Detect lines using the Hough Line Transform
    lines = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=2000, maxLineGap=50)

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.append(line[0])
        elif abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.append(line[0])

    clustered_horizontal = cluster_lines(full_paper, horizontal_lines, orientation='horizontal', threshold=10)
    clustered_vertical = cluster_lines(full_paper, vertical_lines, orientation='vertical', threshold=10)
    clustered_lines = clustered_horizontal + clustered_vertical



    # show_images([closed_edges], ["closed edges"])

    # showLines(full_paper, horizontal_lines, vertical_lines, lines, 0)
    # showLines(full_paper, clustered_horizontal, clustered_vertical, clustered_lines, 1)

    return clustered_horizontal, clustered_vertical

def showLines(full_paper, horizontal_lines, vertical_lines, lines, clustered = 0):
    horizontal = full_paper.copy()
    vertical = full_paper.copy()
    output = full_paper.copy()

    if horizontal_lines is not None:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(horizontal, (x1, y1), (x2, y2), (0, 255, 0), 10)

    if vertical_lines is not None:
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(vertical, (x1, y1), (x2, y2), (0, 255, 0), 10)

    # Draw detected lines
    if lines is not None:
        for line in lines:
            if (clustered == 0):
                x1, y1, x2, y2 = line[0]
            else:
                x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 10)
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

def cluster_lines(full_paper, lines, orientation='horizontal', threshold=10, shifting = 2):
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
                # clustered_lines.append([x_start, avg_y, x_end, avg_y])
                clustered_lines.append([0, avg_y, full_paper.shape[1] - 1, avg_y])
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
                # clustered_lines.append([avg_x, y_start, avg_x, y_end])
                clustered_lines.append([avg_x, 0, avg_x, full_paper.shape[0] - 1])
                cluster = [line]
    
    # Merge the last cluster
    if cluster:
        if orientation == 'horizontal':
            avg_y = int(np.mean([l[1] for l in cluster])) - shifting
            x_start = min(l[0] for l in cluster)
            x_end = max(l[2] for l in cluster)
            # clustered_lines.append([x_start, avg_y, x_end, avg_y])
            clustered_lines.append([0, avg_y, full_paper.shape[1] - 1, avg_y])
        else:
            avg_x = int(np.mean([l[0] for l in cluster])) - shifting
            y_start = min(l[1] for l in cluster)
            y_end = max(l[3] for l in cluster)
            # clustered_lines.append([avg_x, y_start, avg_x, y_end])
            clustered_lines.append([avg_x, 0, avg_x, full_paper.shape[0] - 1])
    
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
            #show_images([cell], [f"{i}, {j}"])
        cells.append(row_cells)
    
    print(f"Extracted {len(cells) * len(cells[0])} cells.")
    return cells
    

def predictCells(cells, digits_models, symbols_models, selected_method):
    for row_cells in cells:
        for i in range(len(row_cells)):
            height, width = row_cells[i].shape[:2]
            start_x = (width - 150) // 2
            start_y = (height - 150) // 2
            end_x = start_x + 150
            end_y = start_y + 150

            # Crop the image
            cropped_cell = row_cells[i][start_y:end_y, start_x:end_x]
            show_images([cropped_cell],["cropped cell"])
            if (i == 0):
                predictID(row_cells[i], selected_method)
            elif (i == 3):
                print(f"The digit predicted is: {predict_digit(cropped_cell, digits_models)}")
            elif (i > 3):
                print(f"The symbol predicted is: {predict_symbol(cropped_cell, symbols_models)}")


def predictID(img, selected_method):
    if (selected_method == "OCR"):
        print ("OCR is not installed :(")
        # print(f"The ID predicted is: {ocr_pytesseract_number_extraction_default(row_cells[i])}")
    else:
        show_images([img], ["img"])
        id_contours_img = img.copy()

        edges = get_edges(img)
        edges = borderTheImage(edges, 12, 12, 12, 12)
        id_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # paper_rectContours = rectContour(paper_contours)
        cv2.drawContours(id_contours_img, id_contours, -1, (0, 255, 0), 1)

        show_images([id_contours_img], ["id contours"])

        x, y, w, h = cv2.boundingRect(id_contours[0])
        questions = img[y:y+h, x:x+w] 

        # SHOWING THE IMAGES FOR CLARITY 

        # show_images([edges, closed_paper, full_paper_contours, full_paper_largest_contour, questions], ["edges","closed", "contours", "largest_contour", "questions"])
        show_images([edges, questions], ["edges","questions"])


############################################# CLASSIFICATON FUNCTIONS ##################################

def extract_hog_features(img):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from the input image.
    This involves resizing the image to a fixed size, dividing it into cells and blocks, 
    computing gradient histograms, and flattening the result into a feature vector. 
    The extracted features are useful for machine learning tasks like image classification and detection.
    """
    target_img_size = (128, 128)
    img = cv2.resize(img, dsize=target_img_size)
    win_size = (128, 128) 
    cell_size = (16, 16) #Divides the window into smaller cells (4x4 pixels per cell).
    block_size_in_cells = (8, 8) 

    # divides window to blocks then blocks to cells
    block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0]) #Determines how much the block moves at each step (4x4 pixels, matching cell size).
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten() #This array is used as input for machine learning models.

def load_model():
    # Load the digits models dictionary
    loaded_digits_models = load('digits_models.joblib')
    print('Loaded digits models from digits_models.joblib')
    print(loaded_digits_models)

    # Load the symbols models dictionary
    loaded_symbols_models = load('symbols_models.joblib')
    print('Loaded symbols models from symbols_models.joblib')

    # Return the loaded models
    return loaded_digits_models, loaded_symbols_models
    

    
def predict_symbol(img, symbols_models):

    test_features=extract_hog_features(img)
    predicted_symbol=symbols_models['SVM'].predict([test_features])
    return predicted_symbol



def predict_digit(img, digits_models):
    test_features=extract_hog_features(img)
    predicted_digit=digits_models['SVM'].predict([test_features])
    predicted_digit_num = ord(predicted_digit[0].lower()) - ord('a') + 1
    
    return predicted_digit_num




########################################## OCR FUNCTIONS ##############################################


def ocr_pytesseract_number_extraction_default(image):
    # Open the image using Pillow
    #you can remove the config to detect the text if you want but we only using it for digits detection
    extracted_text = pytesseract.image_to_string(image, config='--psm 6 -c tessedit_char_whitelist=0123456789')
    return extracted_text

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