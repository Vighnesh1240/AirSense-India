from typing import Dict, List


def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def get_aqi_info(aqi: float) -> Dict[str, str]:
    if aqi <= 50:
        return {
            "category": "Good",
            "risk": "Low",
            "message": "Air quality is considered good."
        }
    if aqi <= 100:
        return {
            "category": "Satisfactory",
            "risk": "Low to Moderate",
            "message": "Air quality is generally acceptable."
        }
    if aqi <= 200:
        return {
            "category": "Moderate",
            "risk": "Moderate",
            "message": "Sensitive individuals may experience discomfort."
        }
    if aqi <= 300:
        return {
            "category": "Poor",
            "risk": "High",
            "message": "Prolonged exposure may affect health."
        }
    if aqi <= 400:
        return {
            "category": "Very Poor",
            "risk": "Very High",
            "message": "Health effects may occur with prolonged exposure."
        }
    return {
        "category": "Severe",
        "risk": "Severe",
        "message": "Air quality presents a serious health concern."
    }


def find_dominant_pollutant(values: Dict[str, float]) -> str:
    """
    Simple concentration-based indicator, not an official AQI sub-index.
    It is used only for UI explanation.
    """
    positive = {
        k: max(float(v), 0.0)
        for k, v in values.items()
        if v is not None
    }
    if not positive:
        return "Unavailable"
    return max(positive, key=positive.get)


def get_recommendations(aqi: float) -> List[str]:
    if aqi <= 50:
        return [
            "Normal outdoor activities are generally appropriate.",
            "Continue routine air-quality awareness."
        ]
    if aqi <= 100:
        return [
            "Outdoor activity is generally acceptable.",
            "Sensitive individuals should monitor symptoms."
        ]
    if aqi <= 200:
        return [
            "Sensitive individuals should reduce prolonged outdoor exertion.",
            "Consider limiting exposure during visible pollution episodes."
        ]
    if aqi <= 300:
        return [
            "Reduce prolonged or strenuous outdoor activity.",
            "Sensitive groups should consider limiting outdoor exposure."
        ]
    if aqi <= 400:
        return [
            "Avoid prolonged outdoor exertion where possible.",
            "Sensitive groups should minimize outdoor exposure."
        ]
    return [
        "Avoid prolonged outdoor exposure.",
        "Follow local public-health and pollution advisories."
    ]
