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
    stat = False
    if confidence <= 0.9:
        plate_ = plate
        mapping = {
            '813ZV': 'A8813ZV',
            'A8120ZW': 'A8720ZW',
            'A9792YC': 'D9792YC',
            'AG8229JT': 'AG8229UT',
            'AG8854RD': 'AG8854RD',
            'AG9104AA': 'AG9104AA',
            'AG9314UR': 'AG9314UR',
            'B0122MY': 'B9112BYW',
            'B046RO': 'B9346KOQ',
            'B0489KDC': 'B9489KDC',
            'B0653UEV': 'B9653UEV',
            'B1002KDJ': 'B1002KQJ',
            'B1667KMM': 'B1667KMM',
            'B1667MMM': 'B1667KMM',
            'B1788VMA': 'B1788VMS',
            'B2128YX': 'B9112BYW',
            'B248OUT': 'B9248UYT',
            'B283TTFS': 'B2831TFS',
            'B2903BYZ': 'B2903BYZ',
            'B3087KCB': 'B9087KCB',
            'B3112YYV': 'B9112BYW',
            'B3122UY': 'B9112BYW',
            'B3122YV': 'B9112BYW',
            'B3240UT': 'B9248UYT',
            'B3248OT': 'B9248UYT',
            'B3248OYT': 'B9248UYT',
            'B3445KC': 'B9445KCA',
            'B3950CCC': 'B9950CUC',
            'B3950CUC': 'B9950CUC',
            'B434TTE': 'B9507TEI',
            'B644ADP': 'B9346KOQ',
            'B706FFF': 'B9706FRV',
            'B706FRF': 'B9706FRV',
            'B8063YV': 'B9663YV',
            'B8112BV': 'B9112BYW',
            'B8112YV': 'B112BYW',
            'B8122BU': 'B9112BYW',
            'B84400': 'B9346KOQ',
            'B8724AAF': 'BE8724AMF',
            'B9012UAA': 'B9012UAS',
            'B9030KEU': 'B9050KEU',
            'B9031JRO': 'B9031JRO',
            'B9031JRQ': 'B9031JRO',
            'B9042CXR': 'B9042CXR',
            'B9050EEU': 'B9050KEU',
            'B9054KCF': 'B9054KCF',
            'B9081KYW': 'B9081KYW',
            'B9102CDA': 'B9102CQA',
            'B9112BV': 'B9112BYW',
            'B9112YYW': 'B9112BYW',
            'B9122BV': 'B9112BYW',
            'B9122YU': 'B9122BYW',
            'B9122YV': 'B9112BYW',
            'B9128SEM': 'B9128SEU',
            'B9128UX': 'B9112BYW',
            'B9175FEU': 'B9175FEU',
            'B9175MO': 'B9775KAQ',
            'B9195KEU': 'B9195KEU',
            'B9205OYW': 'B9205UYW',
            'B9205UYW': 'B9205UYW',
            'B9207K': 'B9207KAU',
            'B9217FXX': 'B9217FXX',
            'B9240UYT': 'B9249UYT',
            'B9241UYT': 'B9241UYT',
            'B9241VT': 'B9241UYT',
            'B9241VTT': 'B9241UYT',
            'B9241VYT': 'B9241UYT',
            'B9241WFV': 'B9241VRV',
            'B9241WRV': 'B9241VRV',
            'B9243UTT': 'B9243UYT',
            'B9243UYT': 'B9243UYT',
            'B9243VTT': 'B9243UYTT',
            'B9243VYT': 'B9243UYT',
            'B9244JYT': 'B9243UYT',
            'B9244VYT': 'B9243UYT',
            'B9244YTT': 'B9243UYT',
            'B9248OMT': 'B9248UYT',
            'B9248OYT': 'B9248UYT',
            'B9248UJT': 'B9248UYT',
            'B9248UT': 'B9248UYT',
            'B9249VYT': 'B9249UYT',
            'B924JUTT': 'B9243UYT',
            'B9250WXR': 'B9250WXR',
            'B9263UYT': 'B9263UYT',
            'B9264QY': 'B9264UYT',
            'B9264UY': 'B9264UYT',
            'B9264VY': 'B9264UYT',
            'B9264YY': 'B9264UYT',
            'B9265SCO': 'B9265SCD',
            'B9267UY': 'B9267UYT',
            'B9267UYT': 'B9267UYT',
            'B9272YXR': 'B9272VXR',
            'B928OYTT': 'B9248UYT',
            'B9297WEK': 'B9297UEK',
            'B9304UTX': 'B9304UIX',
            'B9314KOB': 'B9314KQB',
            'B9340FCI': 'B9340FCL',
            'B9340FCJ': 'B9340FCL',
            'B9340FCL': 'B9340FCL',
            'B9344KCG': 'B9344KCG',
            'B9346OD': 'B9346KOQ',
            'B9346OO': 'B9346KOQ',
            'B9346OZ': 'B9346KOQ',
            'B9346TZZ': 'B9346TEZ',
            'B9348TWW': 'B9348TEW',
            'B9401CQ': 'B9401CQA',
            'B9408CQC': 'B9408CQC',
            'B9438KXV': 'B9438KXV',
            'B9438NXV': 'B9438KXV',
            'B9438WWV': 'B9438KXV',
            'B9438WXV': 'B9438KXV',
            'B9438XXV': 'B9438KXV',
            'B944ROB': 'B9346KOQ',
            'B944ZXC': 'B9344KCG',
            'B9486TYV': 'B9486TYV',
            'B9487EC': 'B9487KDC',
            'B9487KDC': 'B9487KDC',
            'B9487WC': 'B9487KDC',
            'B9489KDC': 'B9489KDC',
            'B9489XDC': 'B9489KDC',
            'B9499SYK': 'B9499SYK',
            'B950KDC': 'B9501KDC',
            'B9511BXR': 'B9511BXR',
            'B9515CCC': 'B9515CQC',
            'B9518TV': 'B9518TXV',
            'B9541KDD': 'B9531KDD',
            'B9600ADC': 'B9600KDC',
            'B9600KDC': 'B9600KDC',
            'B9645BY': 'B9645DY',
            'B9654UEV': 'B9654UEV',
            'B9663YK': 'B9663YV',
            'B9663YV': 'B9663YV',
            'B9667UY': 'B9267UYT',
            'B966UYY': 'B9264UYT',
            'B9704SEU': 'B9704TEU',
            'B9706FFV': 'B8706FRV',
            'B9706FRV': 'B9706FRV',
            'B9718UWX': 'B9718UWX',
            'B9718WWX': 'B9718UWX',
            'B9727KQB': 'B9727KQB',
            'B9737KQB': 'B9737KQB',
            'B9750KYU': 'B9758KYU',
            'B9772JWW': 'B9772UWX',
            'B9775KQ': 'B9775KAQ',
            'B9781QIY': 'B9781UIY',
            'B9788WX': 'B9718UWX',
            'B9789UWX': 'B9789UWX',
            'B9793QIY': 'B9793UIY',
            'B9799UWX': 'B9789UWX',
            'B9834YK': 'B9857YK',
            'B9839QQ': 'B9839UQA',
            'B9849FAP': 'B9849FAP',
            'B9857YK': 'B9857YK',
            'B9871UEJ': 'B9871UEJ',
            'B9900KOA': 'B9900KQA',
            'B9903CCQ': 'B9903TCQ',
            'B9903TCQ': 'B9903TCQ',
            'B9936TYZ': 'B9936TYZ',
            'B9950UWX': 'B9950UWX',
            'B9965XC': 'B9965KCA',
            'B9966FHH': 'B9926FEH',
            'B9968EEO': 'B9968TEU',
            'B9968TEU': 'B9968TEU',
            'B9998CQ': 'B9998CQA',
            'B9C4CCXR': 'B9042CXR',
            'BA8044AB': 'BA8044AB',
            'BB8350FM': 'BE8960AA',
            'BB950CCC': 'B9950CUC',
            'BC044BN': 'BE9840BN',
            'BC4AXU': 'B9346KOQ',
            'BE0334J': 'BE8284OU',
            'BE234CC': 'BE8284OU',
            'BE276NU': 'BE8276NU',
            'BE8040AD': 'BE8040AUD',
            'BE8040FM': 'BE8040FM',
            'BE8099AM': 'BE8099AMB',
            'BE8140AJ': 'BE8140AMJ',
            'BE8140J': 'BE8140AMJ',
            'BE8148J': 'BE8148AMJ',
            'BE8149E': 'BE8149GBA',
            'BE8156MJ': 'BE8156AMJ',
            'BE8177E': 'BE8177AUE',
            'BE8189AA': 'BE8169AUA',
            'BE8192CS': 'BE8192CS',
            'BE8196AA': 'BE8196AUA',
            'BE8196U': 'BE8196AUA',
            'BE8209BK': 'BE8209BK',
            'BE8227BT': 'BE8227BT',
            'BE8276NU': 'BE8276NU',
            'BE8276NV': 'BE8276NU',
            'BE8300D': 'BE8320AUD',
            'BE8311BV': 'BE8311BN',
            'BE8323CP': 'BE8323CP',
            'BE8373BU': 'BE8373BU',
            'BE8373GB': 'BE8373BU',
            'BE8387FU': 'BE8387FU',
            'BE8446MM': 'BE8446AMF',
            'BE8483U': 'BE8483AU',
            'BE8509DU': 'BE8509DU',
            'BE8598AA': 'BE8598GBA',
            'BE8598BA': 'BE8598GBA',
            'BE8622AD': 'BE8642AUD',
            'BE8695AR': 'BE8695AR',
            'BE8695R': 'BE8695AR',
            'BE8723DL': 'BE8723DL',
            'BE8723DM': 'BE8723DL',
            'BE8771L': 'BE8771AML',
            'BE8771ML': 'BE8771AML',
            'BE8775AL': 'BE8775AML',
            'BE8788AA': 'BE8178AUA',
            'BE8864YM': 'BE8964YM',
            'BE8868C': 'BE8868AUC',
            'BE928IAL': 'BE8929AML',
            'BE9571CE': 'BE9571CE',
            'BE9810BN': 'BE9810BN',
            'BE9840BN': 'BE9840BN',
            'BE9963YU': 'BE9963YU',
            'BE9999AL': 'BE8929AML',
            'BG440O': 'B9346KOQ',
            'BG8744AF': 'BE8724AMF',
            'BLB72S': 'BE8284OU',
            'BP835OMI': 'BE8960AA',
            'BT87KDC': 'B9487KDC',
            'BV249VYT': 'B9249UYT',
            'BW276NU': 'BE8276NU',
            'D9195KEU': 'B9195KEU',
            'D95542Y': 'D9542YC',
            'DI5422YC': 'D9542YC',
            'DK8858AA': 'DK8858WA',
            'DP7622YC': 'D9762YC',
            'DP9776YC': 'D9776YC',
            'E19113I': 'E9113F',
            'E8961KK': 'E8961KV',
            'F1745FFE': 'F9745FE',
            'F8648GO': 'F8648GO',
            'F8735BY': 'D8735BY',
            'F9411FG': 'F9411FG',
            'F9744FF': 'F9744FE',
            'F9745FE': 'F9745FE',
            'F9745FF': 'F9745FE',
            'G8229JT': 'AG8229UT',
            'G8229UT': 'AG8229UT',
            'G8854RD': 'AG8854RD',
            'HV332UF': 'G9552OF',
            'L8003UUD': 'L8003UUD',
            'L8009UUU': 'L8009UUD',
            'M1417XYC': 'W1417XI',
            'N8375HN': 'F8375HN',
            'N8426UQ': 'N8426UQ',
            'N8720ZW': 'A8720ZW',
            'P550YYC': 'D9530YC',
            'PP500YT': 'B9250UYT',
            'S0165UN': 'S8165UN',
            'S998BIE': 'B9978BAE',
            'T0467AB': 'T9467AB',
            'T1091Z': 'T1091AZ',


            
        }
        
        if plate_ in mapping:
            plate_ = mapping[plate]
            stat = True
        else:
            plate_ = plate
    else:
        plate_ = plate
        
    return plate_, stat

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