import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import openpyxl
from openpyxl import Workbook
from bubble import *

class FancyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fancy GUI")
        self.root.geometry("600x400")
        
        # Create tabs
        self.tab_control = ttk.Notebook(root)
        
        # Create Bubble Sheet Tab
        self.bubble_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.bubble_tab, text="Bubble Sheet")
        self.create_bubble_tab()
        
        # Create Grades Sheet Tab
        self.grades_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.grades_tab, text="Grades Sheet")
        self.create_grades_tab()
        
        self.tab_control.pack(expand=1, fill="both")

    def create_bubble_tab(self):
        """Create UI elements for the Bubble Sheet tab."""
        title = ttk.Label(self.bubble_tab, text="Bubble Sheet Processor", font=("Helvetica", 16))
        title.pack(pady=20)
        
        upload_button = ttk.Button(self.bubble_tab, text="Upload Images", command=self.process_bubble_sheet_images)
        upload_button.pack(pady=10)
        
        self.bubble_result = tk.Text(self.bubble_tab, height=10, width=50)
        self.bubble_result.pack(pady=10)

    def create_grades_tab(self):
        """Create UI elements for the Grades Sheet tab."""
        title = ttk.Label(self.grades_tab, text="Grades Sheet Processor", font=("Helvetica", 16))
        title.pack(pady=20)
        
        upload_button = ttk.Button(self.grades_tab, text="Upload Image", command=self.upload_single_image)
        upload_button.pack(pady=10)
        
        self.grades_result = tk.Text(self.grades_tab, height=10, width=50)
        self.grades_result.pack(pady=10)

    def process_bubble_sheet_images(self):
        """Handle multiple image uploads for Bubble Sheet and process them."""
        # Ask user to select multiple image files
        image_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not image_paths:
            messagebox.showinfo("Info", "No images selected.")
            return
        
        # Ask user to select a single txt file
        txt_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text Files", "*.txt")]
        )
        if not txt_path:
            messagebox.showinfo("Info", "No text file selected.")
            return

        # Initialize Excel workbook and sheet
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Student Results"
        sheet.append(["ID"] + [f"Q{i+1}" for i in range(100)])  # Header

        accuracy = 0
        num_of_processed_files = 0
        results = []

        for file_path in image_paths:
            file_name = file_path.split("/")[-1]  # Extract the filename
            
            img = cv2.imread(file_path)  # Read the image
            if img is not None:
                try:
                    # Replace this with your solve_bubble_sheet logic
                    currecnt_score, id_number, answers = solve_bubble_sheet(img, txt_path)
                    sheet.append([id_number] + answers)
                    results.append(f"Processed: {file_name} with ID {id_number}")
                    accuracy += currecnt_score
                    num_of_processed_files += 1
                except Exception as e:
                    results.append(f"Error processing {file_name}: {e}")
            else:
                results.append(f"Image {file_name} not found!")

        # Save Excel workbook
        workbook.save("Bubble_Sheet_Student_Results.xlsx")
        results.append(f"Overall accuracy: {accuracy / num_of_processed_files if num_of_processed_files else 0:.2f}")
        
        # Display results in GUI
        self.bubble_result.delete(1.0, tk.END)
        self.bubble_result.insert(tk.END, "\n".join(results))
    
    def upload_single_image(self):
        """Handle single image upload for Grades Sheet."""
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            messagebox.showinfo("Info", "No image selected.")
            return
        
        # Simulate processing (replace with your processing logic)
        result = f"Processed: {file_path}"
        
        self.grades_result.delete(1.0, tk.END)
        self.grades_result.insert(tk.END, result)


# Main Application
if __name__ == "__main__":
    root = tk.Tk()
    app = FancyGUI(root)
    root.mainloop()
