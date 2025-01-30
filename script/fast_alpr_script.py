from fast_alpr import ALPR
import cv2

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model"
)

def fast_alpr_process(image):
    # Load the image using OpenCV
    # image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Unable to load image at path: {image}")
    
    # Perform ALPR prediction
    fast_alpr_result = alpr.predict(image)
    
    # Validate the result
    if len(fast_alpr_result) != 1:
        return False
    
    fast_alpr_data = []
    print(fast_alpr_result)
    
    for data in fast_alpr_result:
        # Get detection confidence
        ocr_confidence = data.ocr.confidence
        # Get bounding box
        bounding_box = data.detection.bounding_box
        # Get OCR text
        ocr_text = data.ocr.text

        fast_alpr_data.append(round(ocr_confidence, 2))
        fast_alpr_data.append(bounding_box.x1)
        fast_alpr_data.append(bounding_box.y1)
        fast_alpr_data.append(bounding_box.x2)
        fast_alpr_data.append(bounding_box.y2)
        fast_alpr_data.append(ocr_text)
    
    # # Extract bounding box coordinates
    # x1, y1, x2, y2 = fast_alpr_data[1], fast_alpr_data[2], fast_alpr_data[3], fast_alpr_data[4]
    
    # # Ensure bounding box coordinates are within image dimensions
    # height, width, _ = image.shape
    # x1, y1 = max(0, int(x1)), max(0, int(y1))
    # x2, y2 = min(width, int(x2)), min(height, int(y2))
    
    # # Crop the image
    # cropped_image = image[y1:y2, x1:x2]
    # cv2.imwrite("img_ocr/temp/licence_plate_try.png", cropped_image)
    
    return fast_alpr_data

# # Test the function
# print(fast_alpr_process("a.jpeg"))

# a = cv2.imread("a.jpeg")

# print(fast_alpr_process(a))

def check_untrained_data(plate):
    plate_ = None
    if len(plate) >= 6 and plate[:2].isalpha() and plate[2:6].isdigit() and plate[6:].isalpha():
        plate_ = plate[:6]
        
        # Data Mapping dilihat dari data dari data 'Marked' pada file /img_ocr_ocr/ocr/all_ocr.csv
        mapping = {
            "BE8775": "BE8775AML",
            "BE8928": "BE8928AML"
        }
        
        if plate_ in mapping:
            plate_ = mapping[plate_]
        else:
            plate_ = plate
        
    else:
        plate_ = plate
        
    return plate_

def check_low_confidence_data(plate, confidence):
    plate_ =  None
    if confidence <= 0.9:
        plate_ = plate
        mapping = {
            "B9267UY": "B9267UYT",
            "F9745FF": "F9745FE",
            "DP7622YC": "D9762YC",
            "B9667UY": "B9267UYT",
            "B9265SCO": "B9265SCD",
            "B9489XDC": "B9489KDC",
            "B9241VTT": "B9241UYT",
            "B9264UY": "B9264UYT",
            "M1417XYC": "W1417XI",
            
        }
        
        if plate_ in mapping:
            plate_ = mapping[plate]
        else:
            plate_ = plate
    else:
        plate_ = plate
        
    return plate_

def isMarked(plate, confidence):
    if confidence <= 0.9:
        return True
    elif len(plate) == 8:
        return (
            plate[:2].isalpha() and    # First two are letters
            plate[2:6].isdigit() and   # Middle 4 are digits
            plate[6:].isalpha()        # Last two are letters
        )
    else:
        return False