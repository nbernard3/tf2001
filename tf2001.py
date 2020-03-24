
import tensorflow as tf

WGS_84_EARTH_MU = 3.986004418e14


@tf.function
def kepler_propagate(orbit, time):
    semi_major_axis, *other_kepler_elements = tf.split(orbit, [1, 5], 1)
    mean_motion = tf.sqrt(WGS_84_EARTH_MU/semi_major_axis)/semi_major_axis
    rate = tf.concat(
        [tf.zeros(shape=(1, 5), dtype='float64'), mean_motion], axis=1)
    orbit_update = rate*time
    return orbit + orbit_update
