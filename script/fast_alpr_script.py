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
            # "BE8775": "BE8775AML",
            # "BE8928": "BE8928AML"
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
            'B0653UEV': 'B9653UEV',
            'DI5422YC': 'D9542YC',
            'D95542Y': 'D9542YC',
            'B8112BV': 'B9112BYW',
            'BE8209BK': 'BE8209BK',
            'B9706FRV': 'B9706FRV',
            'B9128UX': 'B9112BYW',
            'B9175MO': 'B9775KAQ',
            'B9264QY': 'B9264UYT',
            'B9965XC': 'B9965KCA',
            'B9834YK': 'B9857YK',
            'B9936TYZ': 'B9936TYZ',
            'BC044BN': 'BE9840BN',
            'B0122MY': 'B9112BYW',
            'BE8140J': 'BE8140AMJ',
            'BE9840BN': 'BE9840BN',
            'B9775KQ': 'B9775KAQ',
            'F1745FFE': 'F9745FE',
            'B9263UYT': 'B9263UYT',
            'BE8695R': 'BE8695AR',
            'B9243VYT': 'B9243UYT',
            'B3122YV': 'B9112BYW',
            'B944ROB': 'B9346KOQ',
            'B9903TCQ': 'B9903TCQ',
            'B9903CCQ': 'B9903TCQ',
            'B046RO': 'B9346KOQ',
            'BE9963YU': 'BE9963YU',
            'B9244YTT': 'B9243UYT',
            'B2128YX': 'B9112BYW',
            'HV332UF': 'G9552OF',
            'B9511BXR': 'B9511BXR',
            'B9438WXV': 'B9438KXV',
            'BE8177E': 'BE8177AUE',
            'B950KDC': 'B9501KDC',
            '813ZV': 'A8813ZV',
            'B9314KOB': 'B9314KQB',
            'B9030KEU': 'B9050KEU',
            'B9438KXV': 'B9438KXV',
            'B9793QIY': 'B9793UIY',
            'BE8483U': 'BE8483AU',
            'B9600KDC': 'B9600KDC',
            'B9486TYV': 'B9486TYV',
            'B3087KCB': 'B9087KCB',
            'B9244JYT': 'B9243UYT',
            'B9966FHH': 'B9926FEH',
            'B9718UWX': 'B9718UWX',
            'BE8771L': 'BE8771AML',
            'B9248UJT': 'B9248UYT',
            'BE8868C': 'BE8868AUC',
            'BE9810BN': 'BE9810BN',
            'B9788WX': 'B9718UWX',
            'B9267UYT': 'B9267UYT',
            'BE8148J': 'BE8148AMJ',
            'B9654UEV': 'B9654UEV',
            'B9645BY': 'B9645DY',
            'B9346TZZ': 'B9346TEZ',
            'B9195KEU': 'B9195KEU',
            'BE8509DU': 'BE8509DU',
            'BE8192CS': 'BE8192CS',
            'B9112BV': 'B9112BYW',
            'BE8723DM': 'BE8723DL',
            'DP9776YC': 'D9776YC',
            'G8854RD': 'AG8854RD',
            'G8229JT': 'AG8229UT',
            'B9205OYW': 'B9205UYW',
            'B3122UY': 'B9112BYW',
            'B9727KQB': 'B9727KQB',
            'B9998CQ': 'B9998CQA',
            'S998BIE': 'B9978BAE',
            'F8735BY': 'D8735BY',
            'B9789UWX': 'B9789UWX',
            'P550YYC': 'D9530YC',
            'G8229UT': 'AG8229UT',
            'B9240UYT': 'B9249UYT',
            'BE8300D': 'BE8320AUD',
            'N8426UQ': 'N8426UQ',
            'B8063YV': 'B9663YV',
            'B9401CQ': 'B9401CQA',
            'B9241WRV': 'B9241VRV',
            'B9241WFV': 'B9241VRV',
            'B9243UTT': 'B9243UYT',
            'B9112YYW': 'B9112BYW',
            'B9264VY': 'B9264UYT',
            'BV249VYT': 'B9249UYT',
            'BE8311BV': 'BE8311BN',
            'BE8196U': 'BE8196AUA',
            'D9195KEU': 'B9195KEU',
            'B9054KCF': 'B9054KCF',
            'BE8323CP': 'BE8323CP',
            'BE8276NU': 'BE8276NU',
            'B9205UYW': 'B9205UYW',
            'B9737KQB': 'B9737KQB',
            'T1091Z': 'T1091AZ',
            'B9042CXR': 'B9042CXR',
            'B9122YV': 'B9112BYW',
            'BE8771ML': 'BE8771AML',
            'F8648GO': 'F8648GO',
            'B9241VYT': 'B9241UYT',
            'B9663YK': 'B9663YV',
            'BE8149E': 'BE8149GBA',
            'B9968EEO': 'B9968TEU',
            'B9C4CCXR': 'B9042CXR',
            'BE8446MM': 'BE8446AMF',
            'B9081KYW': 'B9081KYW',
            'B9267UY': 'B9267UYT',
            'F9745FF': 'F9745FE',
            'DP7622YC': 'D9762YC',
            'B9667UY': 'B9267UYT',
            'B9265SCO': 'B9265SCD',
            'B9489XDC': 'B9489KDC',
            'B9241VTT': 'B9241UYT',
            'B9264UY': 'B9264UYT',
            'M1417XYC': 'W1417XI',
            'B9122YU': 'B9122BYW',
            'B9207K': 'B9207KAU',
            'B9272YXR': 'B9272VXR',
            'B9346OD': 'B9346KOQ',
            'B9704SEU': 'B9704TEU',
            'B9839QQ': 'B9839UQA',
            'L8009UUU': 'L8009UUD',
            'B3445KC': 'B9445KCA',
            'B9750KYU': 'B9758KYU',
            'BE276NU': 'BE8276NU',
            'BW276NU': 'BE8276NU',
            'BE8276NV': 'BE8276NU',
            'N8375HN': 'F8375HN',
            'B9346OZ': 'B9346KOQ',
            'B9346OO': 'B9346KOQ',
            'B9718WWX': 'B9718UWX',
            'B3248OT': 'B9248UYT',
            'BE0334J': 'BE8284OU',
            'BE234CC': 'BE8284OU',
            'BLB72S': 'BE8284OU',
            'B3950CUC': 'B9950CUC',
            'BB950CCC': 'B9950CUC',
            'B944ZXC': 'B9344KCG',
            'B283TTFS': 'B2831TFS',
            'B9518TV': 'B9518TXV',
            'BB8350FM': 'BE8960AA',
            'BP835OMI': 'BE8960AA',
            'B9871UEJ': 'B9871UEJ',
            'B9487WC': 'B9487KDC',
            'B928OYTT': 'B9248UYT',
            'B248OUT': 'B9248UYT',
            'BE8099AM': 'BE8099AMB',
            'B9243UYT': 'B9243UYT',
            'B9438WWV': 'B9438KXV',
            'B9175FEU': 'B9175FEU',
            'B3248OYT': 'B9248UYT',
            'BE8723DL': 'BE8723DL',
            'B9487KDC': 'B9487KDC',
            'B9849FAP': 'B9849FAP',
            'BE8156MJ': 'BE8156AMJ',
            'B9968TEU': 'B9968TEU',
            'B9900KOA': 'B9900KQA',
            'B9489KDC': 'B9489KDC',
            'DK8858AA': 'DK8858WA',
            'B9799UWX': 'B9789UWX',
            'A8120ZW': 'A8720ZW',
            'B9344KCG': 'B9344KCG',
            'B9249VYT': 'B9249UYT',
            'B9102CDA': 'B9102CQA',
            'B9340FCI': 'B9340FCL',
            'B9248OYT': 'B9248UYT',
            'BE928IAL': 'BE8929AML',
            'BE9999AL': 'BE8929AML',
            'B706FRF': 'B9706FRV',
            'BE8040FM': 'BE8040FM',
            'B9438XXV': 'B9438KXV',
            'BC4AXU': 'B9346KOQ',
            'B9122BV': 'B9112BYW',
            'B1667MMM': 'B1667KMM',
            'B9340FCJ': 'B9340FCL',
            'B9297WEK': 'B9297UEK',
            'BE8598AA': 'BE8598GBA',
            'BE8189AA': 'BE8169AUA',
            'B9241UYT': 'B9241UYT',
            'B3240UT': 'B9248UYT',
            'B9600ADC': 'B9600KDC',
            'F9411FG': 'F9411FG',
            'BE8598BA': 'BE8598GBA',
            'B9244VYT': 'B9243UYT',
            'E19113I': 'E9113F',
            'B0489KDC': 'B9489KDC',
            'BE8373GB': 'BE8373BU',
            'AG9314UR': 'AG9314UR',
            'B9541KDD': 'B9531KDD',
            'B8112YV': 'B112BYW',
            'BE8373BU': 'BE8373BU',
            'B9050EEU': 'B9050KEU',
            'B924JUTT': 'B9243UYT',
            'BT87KDC': 'B9487KDC',
            'BE8864YM': 'BE8964YM',


            
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
    
def is_start_with_numeric(plate):
    return (
        plate[:1].isdigit()
    )