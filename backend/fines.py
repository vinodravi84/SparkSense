def issue_fine(plate, speed, limit=80):
    if speed > limit:
        return {
            "plate": plate,
            "speed": round(speed, 2),
            "limit": limit,
            "fine_amount": (speed - limit) * 10,
            "reason": "Overspeeding"
        }
    return None
