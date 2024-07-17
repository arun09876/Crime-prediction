import time
import tkinter as tk
from tkinter import messagebox, filedialog
import webbrowser
from PIL import Image as PILImage, ImageTk
import pandas as pd
from tkinterweb import TkinterWeb
import webbrowser

def load_data():
    global loaded_data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        loaded_data = file_path
        message_label.config(text="File loaded successfully")

def open_analyze_window():

    # Update message in root window
    message_label.config(text="Welcome to the Analysis Interface!")

    analyze_window = tk.Toplevel(root)
    analyze_window.title("Analysis")
    analyze_window.configure(bg='#2c3e50')

    # Dictionary mapping option names to image paths
    image_options = {
        "HeatMap": "Analysis/heatmap.png",
        "GridSum": "Analysis/gridsum.png",
        "KDE Plot": "Analysis/kdeplot.png",
        "Yearly Sum": "Analysis/yearlyrollingsum.png",
        "Average Crimes": "Analysis/yearly_averaged_crimes.png",
        "No.of Crimes In Location": "Analysis/no.ofcrimes in location.png"
    }

    selected_image = tk.StringVar()
    selected_image.set("HeatMap")  # Set default selection

    # Function to display selected image
    def display_image():
        nonlocal analyze_window
        image_path = image_options[selected_image.get()]
        img = PILImage.open(image_path)
        img_width, img_height = img.size

        # Resize image if its width or height exceeds 800x600
        max_width = 800
        max_height = 600
        if img_width > max_width or img_height > max_height:
            ratio = min(max_width / img_width, max_height / img_height)
            img = img.resize((int(img_width * ratio), int(img_height * ratio)))

        tk_img = ImageTk.PhotoImage(img)

        # Update image label
        image_label.config(image=tk_img)
        image_label.image = tk_img

        # Adjust window geometry based on image size
        analyze_window.geometry(f"{img.width}x{img.height + 200}")

    analyze_window.after(0, display_image)


    button_frame = tk.Frame(analyze_window, bg="#2c3e50")
    button_frame.pack(side="top", fill="x")

    image_dropdown = tk.OptionMenu(button_frame, selected_image, *image_options.keys())
    image_dropdown.pack(side="left", padx=10, pady=10)

    show_image_button = tk.Button(button_frame, text="Show Image", command=display_image)
    show_image_button.pack(side="left", padx=10, pady=10)

    back_button = tk.Button(button_frame, text="Back", command=analyze_window.destroy)
    back_button.pack(side="left", padx=10, pady=10)

    image_label = tk.Label(analyze_window, bg="white")
    image_label.pack(pady=10, fill="both", expand=True)



def forecast():
    # Show processing message for 3 seconds
    message_label.config(text="Processing forecast...", fg="blue")
    root.update()
    time.sleep(3)

    # Load data from Excel file
    excel_data = pd.read_excel('mean_metrics.xlsx')
    # Convert DataFrame to string
    excel_data_str = excel_data.to_string(index=False)

    # Show data in message label
    message_label.config(text=excel_data_str, fg="black")

    # Open HTML file in the default web browser
    webbrowser.open_new("crime_hotspots2.html")


# Create the main window
root = tk.Tk()
root.title("Crime Prediction System")
root.geometry("700x400")
root.configure(bg='#2c3e50')
root.resizable(False, False)

message_label = tk.Label(root, text="", bg="white", fg="black",height=10, width=60)
message_label.place(relx=0.5, rely=0.05, anchor="n")

button_frame = tk.Frame(root, bg='#2c3e50')
button_frame.place(relx=0.5, rely=0.6, anchor="center")

load_button = tk.Button(button_frame, text="Load Data", command=load_data, bg='#27ae60', fg='white', padx=10, pady=5)
load_button.grid(row=0, column=0, padx=10, pady=5)

analyze_button = tk.Button(button_frame, text="Analysis", command=open_analyze_window, bg='#2980b9', fg='white', padx=10, pady=5)
analyze_button.grid(row=0, column=1, padx=10, pady=5)

forecast_button = tk.Button(root, text="Forecast", command=forecast, bg='#e67e22', fg='white', padx=10, pady=5)
forecast_button.place(relx=0.5, rely=0.8, anchor="center")

root.mainloop()
