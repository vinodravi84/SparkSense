fines_db = []

def add_fine(fine):
    fines_db.append(fine)

def get_fines_by_plate(plate):
    return [f for f in fines_db if f["plate"] == plate]
