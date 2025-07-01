import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def read_plate_text(image):
    try:
        results = reader.readtext(image)
        # Filter by length: typical plate is 6-12 characters
        plates = [res[1] for res in results if 6 <= len(res[1]) <= 12]
        return plates[0] if plates else None
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None
