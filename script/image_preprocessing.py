import cv2
from PIL import Image, ImageDraw, ExifTags, UnidentifiedImageError
import os
# from .csv_process import data_photo_uploaded
import csv
import base64
from io import BytesIO

folder_upload = 'img_ocr/upload/'
csv_data_photo_uploaded = 'img_ocr/upload.csv'

def change_image_orientation_to_verical(image, action, datetime):
    """
    Preprocess an image and save it to a directory with the given action and time string.
    :param image: PIL image object or file-like object
    :param action: string, either 'masuk' or 'keluar'
    :param datetime: string, format 'YYYY-MM-DD HH:MM:SS'
    :return: numpy array of the image
    """
    if image is None:
        return False

    try:
        # Open image if it's a file-like object, or verify it's already an Image
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.open(image)
        
        # Force loading to detect any issues
        pil_image.load()
        
    except UnidentifiedImageError:
        print("UnidentifiedImageError: The file is not a valid image.")
        return False
    except AttributeError as e:
        print(f"AttributeError: {e}")
        return False
    
    # Save image in "folder_upload/date/
    folder_uploaded = "img_ocr/temp/"
    
    # Check folder_upload
    if not os.path.exists(folder_uploaded):
        os.makedirs(folder_uploaded)
    
    # Extract the file extension
    global file_extension
    file_extension = os.path.splitext(image.filename)[1]
    
    # Construct the new filename with datetime
    filename = f"{"temp"}.png"
    
    # Rotate image if needed based on EXIF orientation
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = pil_image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation == 8:
                pil_image = pil_image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Handle cases where the image doesn't have EXIF data
        pass
    
    # Save the image
    original_path = os.path.join(folder_uploaded, filename)
    pil_image.save(original_path)
    
    # Read the image
    image = cv2.imread(original_path)
    
    # Update the CSV
    # data_photo_uploaded(csv_data_photo_uploaded, original_path, datetime, action)
    
    return image

def crop_and_save_image(image, x1, y1, x2, y2, action, ocr_result, confidence, datetime, date, time, marker):
    # image, x1, y1, x2, y2, action, ocr_result, confidence, datetime, date, time
    """
    1. Save Cropped Image to img_ocr/ocr/date/filename
       filename = filename_action_ocrresult_date_time_.extension
       save to csv in img_ocr/ocr/date/ocr_date.csv
       path,filename,action,ocr_result,confidence,datetime
       use save_photo_processed()
    2. Save Original Image to img_ocr/upload/date/filename
       filename = filename_action_date_time_.extension
       save to csv in img_ocr/upload/upload_ocr_date.csv
       path,filename,action,datetime
       use save_photo_uploaded()
    """
    # Ensure bounding box coordinates are within image dimensions
    height, width, _ = image.shape
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    
    # Crop the image
    cropped_image = image[y1:y2, x1:x2]
    
    if marker == "MARKED":
        save_marked_image(cropped_image, date, time, ocr_result)
    
    # Save the cropped image
    processed_path = save_photo_processed(
        cropped_image,
        action,
        ocr_result,
        confidence,
        date,
        time,
        datetime,
        marker
    )
    
    # Save original image
    uploaded_path = save_photo_uploaded(
        image, 
        action, 
        date, 
        time,
        datetime,
        marker
    )
    
    return processed_path, uploaded_path

def save_marked_image(cropped_image, date, time, ocr_result):
    # Create directories if they don't exist
    marked_dir = f"img_ocr/marked/{date}"
    os.makedirs(marked_dir, exist_ok = True)
    
    # Generate unique filename
    filename = f"{date}_{time}_{ocr_result}.png"
    full_path = os.path.join(marked_dir, filename)
    
    # Save cropped image
    cv2.imwrite(full_path, cropped_image)
    
    # Log to CSV
    csv_path = os.path.join(marked_dir, f"marked_ocr_{date}.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header if file is new
        if not csv_exists:
            csv_writer.writerow(['Path', 'OCR'])

        # Write image details
        csv_writer.writerow([
            full_path,
            ocr_result
        ])
    
def save_photo_uploaded(image, action, date, time, datetime, marker):
    # Create directories if they don't exist
    uploaded_dir = f"img_ocr/upload/{date}"
    os.makedirs(uploaded_dir, exist_ok = True)
    
    # Generate unique filename
    filename = f"{action}_{date}_{time}.png"
    full_path = os.path.join(uploaded_dir, filename)
    
    # Save image
    cv2.imwrite(full_path, image)
    
    # Log to CSV
    csv_path = os.path.join(uploaded_dir, f"upload_ocr_{date}.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header if file is new
        if not csv_exists:
            csv_writer.writerow(['Path', 'Action', 'Datetime', 'Marker'])

        # Write image details
        csv_writer.writerow([
            full_path,
            action, 
            datetime,
            marker
        ])
    
    # Log all Data to CSV
    all_csv_path = "img_ocr/upload/all_upload.csv"
    all_csv_exists = os.path.exists(all_csv_path)
    
    with open(all_csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not all_csv_exists:
            csv_writer.writerow(['Path', 'Action', 'Datetime', 'Marker'])
        
        # Write image details
        csv_writer.writerow([
            full_path,
            action, 
            datetime,
            marker
        ])

    return full_path

def save_photo_processed(cropped_image, action, ocr_result, confidence, date, time, datetime, marker):
    # Create directories if they don't exist
    processed_dir = f"img_ocr/ocr/{date}"
    os.makedirs(processed_dir, exist_ok = True)
    
    # Generate unique filename
    filename = f"{ocr_result}_{confidence}_{action}_{date}_{time}.png"
    full_path = os.path.join(processed_dir, filename)
    
    # Save cropped image
    cv2.imwrite(full_path, cropped_image)
    
    # Log to CSV
    csv_path = os.path.join(processed_dir, f"ocr_{date}.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header if file is new
        if not csv_exists:
            csv_writer.writerow(['Path', 'Action', 'OCR Result', 'Confidence', 'Datetime', 'Marker'])

        # Write image details
        csv_writer.writerow([
            full_path,
            action,
            ocr_result, 
            confidence, 
            datetime,
            marker
        ])
        
    # Log all Data to CSV
    all_csv_path = "img_ocr/ocr/all_ocr.csv"
    all_csv_exists = os.path.exists(all_csv_path)
    
    with open(all_csv_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header if file is new
        if not all_csv_exists:
            csv_writer.writerow(['Path', 'Action', 'OCR Result', 'Confidence', 'Datetime', 'Marker'])
        
        # Write image details
        csv_writer.writerow([
            full_path,
            action,
            ocr_result, 
            confidence, 
            datetime,
            marker
        ])
    
    return full_path

# Assuming you have an image loaded with cv2
# image = cv2.imread('a.jpg')
# a = crop_and_save_image(
#     image, 
#     x1=100, 
#     y1=200, 
#     x2=300, 
#     y2=400, 
#     action='licence_plate', 
#     ocr_result='ABC123', 
#     confidence=0.95,
#     datetime='2025-01-16 12:00:00',
#     date='2025-01-16',
#     time='12-00-00'
# )

# print(a)

def numpy_to_base64(image_np):
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    
    # Save the PIL image to a BytesIO buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    
    # Encode the image to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"