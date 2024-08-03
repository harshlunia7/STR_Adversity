from .fog.fog_degradation import FogDegradation
from .lowlight.lowlight_degradation import LowLightDegradation
from .rain.rain_degradation import RainDegradation
from .snow.snow_degradation import SnowDegradation

class DegradationFactory:
    """Factory class to create degradation instances."""
    @staticmethod
    def get_degradation(degradation_type, degree, image_type):
        if degradation_type == "fog":
            return FogDegradation(degree, image_type)
        elif degradation_type == "lowlight":
            return LowLightDegradation(degree, image_type)
        elif degradation_type == "rain":
            return RainDegradation(degree, image_type)
        elif degradation_type == "snow":
            return SnowDegradation(degree, image_type)
        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

