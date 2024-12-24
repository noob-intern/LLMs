import json

def get_guitar_info(guitar_name: str) -> dict:
    mock_db = {
        "Acoustica Deluxe": {
            "body_material": "Solid Sitka Spruce",
            "neck_material": "Mahogany",
            "pickup_series": "ToneMaster Pickup Series",
            "price": "$2,500",
        },
        "Electric Pro X": {
            "body_material": "Alder",
            "neck_material": "Maple",
            "pickup_series": "ToneMaster Bridge Humbucker",
            "price": "$1,999",
        },
    }
    if guitar_name not in mock_db:
        return {
            "error": f"Guitar '{guitar_name}' not found in database."
        }
    return {
        "guitar_name": guitar_name,
        "details": mock_db[guitar_name]
    }