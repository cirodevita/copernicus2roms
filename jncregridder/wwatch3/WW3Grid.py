import re
import numpy as np


class WW3Grid:
    def __init__(self, inp_file):
        self.inp_file = inp_file
        self.ncols = 0
        self.nrows = 0
        self.minLon = 0
        self.minLat = 0
        self.maxLon = 0
        self.maxLat = 0
        self.lats = []
        self.lons = []

        self.__load()

    def __load(self):
        lines = []
        with open(self.inp_file) as fp:
            line = fp.readline()
            while line:
                if line:
                    line = re.sub(' +', ' ', line.strip().replace("\t"," ")) 
                    if not line.startswith("$"):
                        lines.append(line)
                line = fp.readline()
        
        count = 0
        for line in lines:
            if 'RECT' in line:
                break
            count = count + 1

        parts = lines[count+1].split(" ")
        self.ncols = int(parts[0])
        self.nrows = int(parts[1])

        parts = lines[count+2].split(" ")
        dLon = round(float(parts[0]) / float(parts[2]), 5)
        dLat = round(float(parts[1]) / float(parts[2]), 5)

        parts = lines[count+3].split(" ")
        self.minLon = float(parts[0]) / float(parts[2])
        self.minLat = float(parts[1]) / float(parts[2])

        self.maxLat = self.minLat + dLat * self.nrows
        self.maxLon = self.minLon + dLon * self.ncols

        # Create the latitude array
        self.lats = np.arange(self.minLat, self.maxLat + dLat, dLat)
        self.lats = self.lats[:self.nrows]

        # create the longitude array
        self.lons = np.arange(self.minLon, self.maxLon + dLon, dLon)
        self.lons = self.lons[:self.ncols]
