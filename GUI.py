import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import openpyxl
from openpyxl import Workbook
from bubble import *
from grades_sheet import *

class IntegratedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bubble and Grades Sheet Processor")
        self.root.geometry("600x400")
        
        self.create_main_tab()

    def create_main_tab(self):
        """Create main tab with options for Bubble Sheet and Grades Sheet."""
        self.clear_frame()
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=20)
        
        tk.Label(self.main_frame, text="Choose an Option:", font=("Arial", 14)).pack()
        
        # Bubble Sheet Button
        tk.Button(
            self.main_frame, text="Bubble Sheet", font=("Arial", 12), command=self.create_bubble_tab
        ).pack(pady=10)
        
        # Grades Sheet Button
        tk.Button(
            self.main_frame, text="Grades Sheet", font=("Arial", 12), command=self.create_grades_tab
        ).pack(pady=10)

    def create_bubble_tab(self):
        """Create Bubble Sheet tab."""
        self.clear_frame()
        tk.Label(self.root, text="Bubble Sheet Processor", font=("Arial", 16)).pack(pady=10)
        
        upload_button = tk.Button(
            self.root, text="Upload Bubble Sheet Images", font=("Arial", 12), command=self.process_bubble_sheet_images
        )
        upload_button.pack(pady=10)
        
        self.bubble_result = tk.Text(self.root, height=10, width=50)
        self.bubble_result.pack(pady=10)
        
        # Back button to return to the main page
        back_button = tk.Button(
            self.root, text="Back", font=("Arial", 12), command=self.create_main_tab
        )
        back_button.pack(pady=10)

    def create_grades_tab(self):
        """Create the Grades Sheet tab with OCR and Features + Classifier options."""
        self.clear_frame()
        tk.Label(self.root, text="Choose a Method for Grades Sheet:", font=("Arial", 14)).pack(pady=10)
        
        # Option 1: OCR
        tk.Button(
            self.root, text="OCR", font=("Arial", 12), command=lambda: self.process_grades_sheet("OCR")
        ).pack(pady=5)
        
        # Option 2: Features + Classifier
        tk.Button(
            self.root, text="Features + Classifier", font=("Arial", 12), command=lambda: self.process_grades_sheet("Features + Classifier")
        ).pack(pady=5)

        # Back button to return to the main page
        back_button = tk.Button(
            self.root, text="Back", font=("Arial", 12), command=self.create_main_tab
        )
        back_button.pack(pady=10)

    def process_bubble_sheet_images(self):
        """Handle multiple image uploads for Bubble Sheet and process them."""
        image_paths = filedialog.askopenfilenames(
            title="Select Images", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not image_paths:
            messagebox.showinfo("Info", "No images selected.")
            return

        txt_path = filedialog.askopenfilename(
            title="Select Text File", filetypes=[("Text Files", "*.txt")]
        )
        if not txt_path:
            messagebox.showinfo("Info", "No text file selected.")
            return

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Student Results"
        sheet.append(["ID"] + [f"Q{i+1}" for i in range(100)])
        
        results = []
        accuracy = 0
        num_of_processed_files = 0

        for file_path in image_paths:
            file_name = file_path.split("/")[-1]
            img = cv2.imread(file_path)
            if img is not None:
                try:
                    current_score, id_number, answers = solve_bubble_sheet(img, txt_path)
                    sheet.append([id_number] + answers)
                    results.append(f"Processed: {file_name} with ID {id_number}")
                    accuracy += current_score
                    num_of_processed_files += 1
                except Exception as e:
                    results.append(f"Error processing {file_name}: {e}")
            else:
                results.append(f"Image {file_name} not found!")

        workbook.save("Bubble_Sheet_Student_Results.xlsx")
        results.append(f"Overall accuracy: {accuracy / num_of_processed_files if num_of_processed_files else 0:.2f}")
        
        self.bubble_result.delete(1.0, tk.END)
        self.bubble_result.insert(tk.END, "\n".join(results))

    def process_grades_sheet(self, selected_method):
        """Process the Grades Sheet with the selected method."""
        image_path = filedialog.askopenfilename(
            title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not image_path:
            messagebox.showinfo("Info", "No image selected.")
            return

        try:
            digits_models, symbols_models = load_model()
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Student Results"
            sheet.append(["ID"] + [f"Q{i+1}" for i in range(100)])
            
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (3000, 3000))
            full_paper = getPerspective(img)[100:, :]
            clustered_horizontal, clustered_vertical = getLines(full_paper)
            cells = extractCells(full_paper, clustered_horizontal, clustered_vertical)
            predictCells(cells, digits_models, symbols_models, selected_method, sheet)
            workbook.save("Grades_Sheet.xlsx")
            messagebox.showinfo("Success", f"Grades Sheet processed with {selected_method}. Results saved in Grades_Sheet.xlsx")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def clear_frame(self):
        """Clear the main frame content."""
        for widget in self.root.winfo_children():
            widget.destroy()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedApp(root)
    root.mainloop()
