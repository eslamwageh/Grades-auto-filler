from grades_sheet_functions import *
import ipywidgets as widgets
from IPython.display import display, clear_output
import openpyxl
from openpyxl import Workbook

selected_method = None
digits_models, symbols_models = load_model()

workbook = Workbook()
sheet = workbook.active
sheet.title = "Student Results"

# Write the header
sheet.append(["ID"] + [f"Q{i+1}" for i in range(100)]) 

def program(selected_method = 'Features + Classifier'):
    img = cv2.imread('start.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    img = cv2.resize(img, (3000, 3000))
    full_paper = getPerspective(img)
    full_paper = full_paper[100:, :]
    clustered_horizontal, clustered_vertical = getLines(full_paper)
    cells = extractCells(full_paper, clustered_horizontal, clustered_vertical)
    predictCells(cells, digits_models, symbols_models, selected_method, sheet)
    workbook.save("Grades_sheet.xlsx")


# program('OCR')
