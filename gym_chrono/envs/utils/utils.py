# PyChrono imports
import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.sensor as sens

import math
import numpy as np

# Function from ART for transforming 

class graph(object):
    def __init__(self):
        self.r = self.r = 6378100

    def set_graph(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def gps2cartesian(self, lat, lon, alt):
        x = ((lon - self.lon) * math.pi / 180.0) * \
            (self.r * math.cos(lat * math.pi / 180.0))
        y = ((lat - self.lat) * math.pi / 180.0) * self.r
        z = alt - self.alt
        return x, y, z

    def set_rotation(self, D):
        self.R = np.array([
            [math.cos(D), -math.sin(D)],
            [math.sin(D), math.cos(D)]
        ])
        self.R = np.linalg.inv(self.R)

    def rotate(self, x, y, z):
        position = np.array([
            [x],
            [y]
        ])
        position = self.R @ position

        return position[0][0], position[1][0], z


def SetChronoDataDirectories():
    """
    Set data directory

    This is useful so data directory paths don't need to be changed everytime
    you pull from or push to github. To make this useful, make sure you perform
    step 2, as defined for your operating system.

    For Linux or Mac users:
      Replace bashrc with the shell your using. Could be .zshrc.
      1. echo 'export CHRONO_DATA_DIR=<chrono's data directory>' >> ~/.bashrc
          Ex. echo 'export CHRONO_DATA_DIR=/home/user/chrono/data/' >> ~/.zshrc
      2. source ~/.zshrc

    For Windows users:
      Link as reference: https://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows/
      1. Open the System Properties dialog, click on Advanced and then Environment Variables
      2. Under User variables, click New... and create a variable as described below
          Variable name: CHRONO_DATA_DIR
          Variable value: <chrono's data directory>
              Ex. Variable value: C:\ Users\ user\ chrono\ data\
    """
    from pathlib import Path
    import os

    CONDA_PREFIX = os.environ.get('CONDA_PREFIX')
    CHRONO_DATA_DIR = os.environ.get('CHRONO_DATA_DIR')
    if CONDA_PREFIX and not CHRONO_DATA_DIR:
        CHRONO_DATA_DIR = os.path.join(
            CONDA_PREFIX, 'share', 'chrono', 'data', '')
    if not CHRONO_DATA_DIR:
        CHRONO_DATA_DIR = os.path.join(Path(os.path.dirname(
            os.path.realpath(__file__))).parents[1], 'chrono', 'data', '')
    elif not CHRONO_DATA_DIR:
        raise Exception(
            'Cannot find the chrono data directory. Please verify that CHRONO_DATA_DIR is set correctly.')

    chrono.SetChronoDataPath(CHRONO_DATA_DIR)
    veh.SetDataPath(os.path.join(CHRONO_DATA_DIR, 'vehicle', ''))


def CalcInitialPose(p1: chrono.ChVectorD, p2: chrono.ChVectorD, z=0.1, reversed=0):
    if not isinstance(p1, chrono.ChVectorD):
        raise TypeError
    elif not isinstance(p2, chrono.ChVectorD):
        raise TypeError

    p1.z = p2.z = z

    initLoc = p1

    vec = p2 - p1
    theta = math.atan2((vec % chrono.ChVectorD(1, 0, 0)
                        ).Length(), vec ^ chrono.ChVectorD(1, 0, 0))
    if reversed:
        theta *= -1
    initRot = chrono.ChQuaternionD()
    initRot.Q_from_AngZ(theta)

    return initLoc, initRot


def chVector_to_npArray(v: chrono.ChVectorD):
    if not isinstance(v, chrono.ChVectorD):
        raise TypeError

    return np.array([v.x, v.y, v.z])


def npArray_to_chVector(v: np.ndarray):
    if not isinstance(v, np.ndarray):
        raise TypeError

    return chrono.ChVectorD(v[0], v[1], v[2])
