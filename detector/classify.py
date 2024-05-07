def process_license_plate(plate_number):
    if plate_number is not None:
        try:
            if plate_number[0].isalpha() and plate_number[1:4].isdigit() and plate_number[4:].isalpha():
                return "Passanger vehicle"

            if plate_number[:2].isalpha() and plate_number[2].isdigit():
                return "Bus, road technic or cargo vehicle"

            if plate_number[:3].isdigit() and plate_number[3].isalpha():
                return "Diplomacy"

            if plate_number[0].isalpha() and plate_number[1:].isdigit():
                return "Police"

            if plate_number[:4].isdigit() and plate_number[4:].isalpha():
                return "Armed force"
        except:
            return "Text recognizer returned incorrect number form"
    
    return "Couldn't read the number of region"