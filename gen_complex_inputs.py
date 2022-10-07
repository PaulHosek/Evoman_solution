import numpy as np

sensors = np.zeros(20)
# 1. horizontal dist enemy - player
# 2. vertical dist enemy - player
# 3.
# 19. player facing direction
# 20. enemy facing direction



def is_facing(sensors):
    # how?
    # used distance player-enemy + player facing
    pass
np.random.seed(0)


def euclidian_distance(sensors, loc_player):
    """
    Calculate euclidian distance between player and all projectiles.
    @param sensors:
    @return: np.array of len 8.
    """

    projectile_coords = sensors[3:19]  # get all projectile x and y
    # projectile_coords = sensors
    e_distances = np.empty(8)
    # iterate over coordinate pairs
    for i, loc_proj in enumerate(np.dstack((projectile_coords[:-1:2], projectile_coords[1::2]))):
        e_distances[i] = np.linalg.norm(loc_player-loc_proj,ord=2)
    return e_distances





# test data
my_sensors = np.random.randint(1,200,size=20)
print(my_sensors)
new = euclidian_distance(my_sensors, loc_player=(1,1))
print(new)
