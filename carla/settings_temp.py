from lib import utils
import sys

try:
    carla_egg = utils.get_carla_egg()
    sys.path.append(carla_egg)
except (IndexError, TypeError) as e:
    raise e("Please specify path to Carla .egg file in config/config.ini")

import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

settings = world.get_settings()
settings.no_rendering_mode = True
world.apply_settings(settings)


