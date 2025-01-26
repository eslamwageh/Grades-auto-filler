from bubble_sheet_functions import *
folder_path = "Bubble_sheet/"

def get_choices_number(one_col_of_questions):
        
    blurred = cv2.GaussianBlur(one_col_of_questions, (3, 3), 0.7)
    edges = cv2.Canny(blurred, 50, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertically_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
    #show_images([vertically_closed], ["vertically_closed"])

    questions_contours, _ = cv2.findContours(vertically_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(questions_contours) - 1

def divide_questions(questions):

    blurred = cv2.GaussianBlur(questions, (7, 7), 5)
    edges = cv2.Canny(blurred, 50, 120)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 60))
    horizontally_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # show_images([horizontally_closed], ["horizontally_closed"])
    #now we have multiple colomuns of questions and we need to separate them

    # Find contours of the closed edges (columns)
    contours, _ = cv2.findContours(horizontally_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left to right based on x-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # Extract each column using the bounding boxes of the contours
    columns = []
    for idx, contour in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        column = questions[y:y + h, x:x + w]
        # show_images([column], [f'col{idx}'])
        columns.append(column)
    
    # get the number of questions in each column
    choices_number = get_choices_number(columns[0])

    print(choices_number)  # print the number of choices in each column
    # show_images([columns[0],columns[1],columns[2]],['1','2','3'])

    return columns

def read_correct_answers(file_path):
    with open(file_path, "r") as f:
        correct_answers = [line.strip() for line in f.readlines()]
    return correct_answers


def get_questions_from_column(column_image):
    """
    Given a single column of questions, detect and return each question as an individual image.
    """
    # Get the edges of the column
        
    # show_images([column_image], ["column_image"])

    blurred = cv2.GaussianBlur(column_image, (5, 5), 3)
    edges = cv2.Canny(blurred, 50, 120)

    # Define the structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))

    # Apply the closing operation
    horizontally_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
    # show_images([horizontally_closed], ["horizontally_closed"])

    # Find contours in the dilated image
    contours, _ = cv2.findContours(horizontally_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # invert the questions to make them from 1->15 not 15->1
    contours = contours[::-1]
    # Extract each question based on the contours
    questions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        question_image = column_image[y:y+h, x:x+w]  # Extract the question region
        #show_images([question_image], ["question_image"])
        # Process the question image: dilate and remove the first column
        processed_question, num_of_choices = process_question_image(question_image)
        
        #print(detect_shaded_choice(processed_question))

        # Append the processed question to the list
        questions.append(processed_question)
        # questions.append(question_image)
    
    return questions, num_of_choices


def process_question_image(question_image):
    """
    Process a question image: dilate it horizontally, find contours, 
    remove the first contour, and return the bounding rectangle containing the rest.
    """
    # Get the edges of the question
    blurred = cv2.GaussianBlur(question_image, (3, 3), 0.7)
    edges = cv2.Canny(blurred, 50, 120)

    # Dilate horizontally to group components of the question (bubble options)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))  # Horizontal dilation
    dilated_question = cv2.dilate(edges, kernel, iterations=2)

    # Show the dilated image 
    #show_images([dilated_question], ["dilated_question"])


    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated_question, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours top-to-bottom based on their y-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Remove the first contour (typically contains the numbering or unwanted part)
    if sorted_contours:
        sorted_contours = sorted_contours[:-1]  # Remove the first contour

    # If there are remaining contours, calculate the bounding rectangle around them
    if sorted_contours:
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Extract the rectangle from the original question image
        processed_question = question_image[y_min:y_max, x_min:x_max]
    else:
        # If no valid contours, return an empty image
        processed_question = question_image

    #show_images([processed_question], ["processed_question"])
    return processed_question, len(sorted_contours)  # to return the numberr of questions 



def get_all_questions(questions_img):
    """
    Given the full image of all questions, divide it into columns and then extract each individual question.
    """
    # Divide the image into columns
    columns = divide_questions(questions_img)

    all_questions = []
    for column in columns:
        # Extract individual questions from each column
        column_questions, num_of_choices = get_questions_from_column(column)
        all_questions.extend(column_questions)  # Add the questions from this column to the list
    
    return all_questions, num_of_choices

def detect_shaded_choice(question_image, num_choices=4, id_or_question = 0): # 0 for question as if id the erosion will be different
    """
    Detect the shaded choice in a question image.
    :param question_image: The input image of a single question.
    :param num_choices: Number of answer choices (e.g., 4 for A, B, C, D).
    :return: The index of the shaded choice (0 for A, 1 for B, etc.), or -1 if none is shaded.
    """
    # Convert to grayscale
    #gray = cv2.cvtColor(question_image, cv2.COLOR_BGR2GRAY)
    # show_images([question_image], ["question_image"])

    if(id_or_question == 0):
        # Threshold the image to emphasize shading
        _, binary = cv2.threshold(question_image, 147, 255, cv2.THRESH_BINARY_INV)  # Binary for shaded areas
        # show_images([binary], ["Binary"])

        # Erode the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_choices = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        eroded_choices = cv2.morphologyEx(closed_choices, cv2.MORPH_ERODE, kernel, iterations=9)
        #final_choice = cv2.dilate(eroded_choices, kernel, iterations=14)
    else:
        edges = get_edges(question_image)
        gray = cv2.cvtColor(question_image, cv2.COLOR_RGB2GRAY)
        #show_images([gray], ['gray'])
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_choices = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        eroded_choices = cv2.morphologyEx(closed_choices, cv2.MORPH_ERODE, kernel, iterations=8)
        # show_images([question_image, binary, closed_choices, eroded_choices, edges], ["question_image","binary", "closed choices","Eroded Choices", 'edges'])

    # show_images([closed_choices, eroded_choices], ["closed choices","Eroded Choices"])

    # Split the question into equal regions for each choice
    h, w = eroded_choices.shape
    choice_width = w // num_choices  # Assuming equal width for each choice
    user_answers = []
    total_pixels_per_region = h * choice_width
    #print(total_pixels_per_region)

    for i in range(num_choices):
        # Extract the region for the current choice
        x_start = i * choice_width
        x_end = x_start + choice_width
        choice_region = eroded_choices[:, x_start:x_end]
        

        num_of_white_pixels = np.sum(choice_region == 255)
        #print(num_of_white_pixels)
        if num_of_white_pixels > 0:
            user_answers.append((i, num_of_white_pixels))
    
    
    if len(user_answers) == 1:
        return user_answers[0][0]  # Return the i value directly
    elif len(user_answers) > 1 and id_or_question:   # case of extractring the id we will choose the cell with the largest number of white pixels
        # Find the pair with the largest number of white pixels
        max_pair = max(user_answers, key=lambda x: x[1])
        return max_pair[0]  # Return the i value


    return '#'  # No definitive answer
    
def getID(full_paper):
    full_paper_contours = full_paper.copy()
    full_paper_largest_contour = full_paper.copy()

    edges = get_edges(full_paper)

    # Create a mask with zeros at the borders and ones elsewhere
    height, width = edges.shape
    border_mask = np.zeros((height, width), dtype=np.uint8)
    border_mask[15:-15, 15:-15] = 1  # Leave a 1-pixel border untouched

    # Dilate the edges
    #print(edges.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertically_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=10)
    #show_images([full_paper, vertically_closed], ["full paper", "vertically_closed"])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 15))
    closed = cv2.morphologyEx(vertically_closed, cv2.MORPH_CLOSE, kernel, iterations=2)

    #show_images([closed], ["closed"])

    # Apply the border mask to retain original borders
    closed_paper = np.where(border_mask, closed, edges)

    closed_paper[:15, :] = 0
    closed_paper[-15:, :] = 0
    closed_paper[:, :15] = 0
    closed_paper[:, -15:] = 0

    paper_contours, _ = cv2.findContours(closed_paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # paper_rectContours = rectContour(paper_contours)
    cv2.drawContours(full_paper_contours, paper_contours, -1, (0, 255, 0), 3)

    # Find the most top-left contour based on bounding box position
    most_top_left_contour = None
    min_x = float('inf')
    min_y = float('inf')

    for contour in paper_contours:
        # Get the bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if this contour is more top-left than the current one
        if x < min_x or  y < min_y:
            most_top_left_contour = contour
            min_x = x
            min_y = y

    # Draw the most top-left contour
    cv2.drawContours(full_paper_largest_contour, [most_top_left_contour], -1, (0, 255, 0), 3)

    x, y, w, h = cv2.boundingRect(most_top_left_contour)
    id = full_paper[y:y+h, x:x+w] 

    # SHOWING THE IMAGES FOR CLARITY 
    #show_images([edges, closed_paper, full_paper_contours, full_paper_largest_contour, id], ["edges", "closed", "contours", "most_top_left_contour", "id"])

    return id

def extractID_rows_then_id(id_image):
    edges = get_edges(id_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))
    horizontally_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    #show_images([horizontally_closed],['horizontally_closed'])

    # vertical closing to the the number of digitis in the id 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 30))
    vertically_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    choices_contours, _ = cv2.findContours(vertically_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #show_images([vertically_closed], ['vertically_closed'])

    number_of_id_digits = len(choices_contours)
    print(f'Here is the number of digits of this file{number_of_id_digits}')

    # Find contours of the closed edges (columns)
    contours, _ = cv2.findContours(horizontally_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours top to bottom based on y-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # Initialize the concatenated number
    concatenated_id = ""

    # Extract each row using the bounding boxes of the contours
    for idx, contour in enumerate(sorted_contours):
        if idx == 0:  # Skip the first row as it is the ID
            continue
        x, y, w, h = cv2.boundingRect(contour)
        row = id_image[y:y + h, x:x + w]

        # Process the row to extract a number
        number = detect_shaded_choice(row, number_of_id_digits , 1) 
        
        # Concatenate the number to the ID
        concatenated_id += str(number)
        
        # For debugging, show the row being processed
        #show_images([row], 'row')

        # print(concatenated_id)
    
    return concatenated_id

def solve_bubble_sheet(original_image, correct_answers_file):
    if original_image is not None:  # Check if the image exists
        # Preprocess the image
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (1000, 1400))

        # Get the full paper and questions
        full_paper = getPerspective(img)
        questions = getQuestions(full_paper)
        # show_images([questions], ['questions'])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_questions = clahe.apply(cv2.cvtColor(questions, cv2.COLOR_RGB2GRAY))
        #show_images([questions,equalized_questions,cv2.cvtColor(questions, cv2.COLOR_RGB2GRAY)], ['questions','eq','gr'])
        id_image = getID(full_paper)
        # show_images([id_image], ['id_image'])
        id_number = extractID_rows_then_id(id_image)
        all_questions, num_of_choices = get_all_questions(equalized_questions)
        #print(f"Total number of questions: {len(all_questions)}")

        # Read correct answers from the file
        correct_answers = read_correct_answers(correct_answers_file)
        
        # Initialize score
        score = 0

        answers_to_excel = []

        # Loop through all questions and compare detected answers to correct answers
        print(f'lenght of all questions: {len(all_questions)}')
        print(f'lenght of correct answers: {len(correct_answers)}')

        for idx, question_image in enumerate(all_questions):
            # Detect the student's answer for the current question
            detected_answer = detect_shaded_choice(question_image, num_of_choices)
            
            # Get the correct answer for the current question (if available)
            if idx < len(correct_answers):
                correct_answer = correct_answers[idx]
                #print(f"Detected answer: {detected_answer} for idx{idx + 1}")
                #print(f"Correct answer (index): {ord(correct_answer) - 65}")
                #print('-------------------------------------------')
                if detected_answer != '#' and detected_answer == ord(correct_answer) - 65:  
                    score += 1
                    answers_to_excel.append(1)
                    #print(f"Question {idx + 1}: Correct")
                else:
                    print(f"Question {idx + 1}: Incorrect")
                    answers_to_excel.append(0)
            else:
                print(f"Question {idx + 1}: No correct answer available")

        # Output the final score
        print(f"Total Score: {score} for id: {id_number}") # 
        return score / len(correct_answers), id_number, answers_to_excel
    else:   
        print("Image not found!")
        
    return 0


# img = cv2.imread('Bubble_sheet/29.jpg')
# solve_bubble_sheet(img, 'Bubble_sheet/29.txt')

# import openpyxl
# from openpyxl import Workbook

# # Initialize a workbook and sheet
# workbook = Workbook()
# sheet = workbook.active
# sheet.title = "Student Results"

# # Write the header
# sheet.append(["ID"] + [f"Q{i+1}" for i in range(100)])  # Adjust 100 to a reasonable maximum number of questions

# accuracy = 0
# num_of_processed_files = 0
# # # #Loop through the range of numbers from 1 to 39
# for i in range(1, 40):  # 1 to 39 inclusive
#     if (i == 18): continue  #as this image is not in the images
#     print(f'now processing file {i}')


#     file_name = f"{i}.jpg"  # Generate the file name
#     txt_name = f"{i}.txt" 
#     file_path = folder_path + file_name  # Create the full path
#     txt_path = folder_path + txt_name  # Create the full path for the correct answers file
#     img = cv2.imread(file_path)  # Read the image
#     if img is not None:  # Check if the image exists
#         currecnt_score, id_number, answers = solve_bubble_sheet(img, txt_path)
#         print(f'printed the result of file {i}')

#         # Append to the Excel sheet
#         sheet.append([id_number] + answers)
        
#         accuracy += currecnt_score
#         num_of_processed_files += 1

#     else:
#         print(f"Image {file_name} not found!")
#     #print(detect_shaded_choice(all_questions[0]))
# workbook.save("Bubble_Sheet_Student_Results.xlsx")
# print(f'the overall accuracy = {accuracy / num_of_processed_files}')