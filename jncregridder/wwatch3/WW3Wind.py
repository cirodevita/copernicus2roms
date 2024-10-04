import os
import math
import numpy as np

from jncregridder.util.Interpolator import Interpolator
from jncregridder.wrf.WRFData import WRFData


class WW3Wind:
    def __init__(self, dst, ww3Grid, wrfFilenameOrFilesPath):
        self.dst = dst
        self.ww3Grid = ww3Grid
        self.wrfFiles = []
        self.variables = ["XLAT", "XLONG", "uvmet10"]

        folder = os.path.abspath(wrfFilenameOrFilesPath)
        if os.path.isdir(folder):
            listOfFiles = sorted(os.listdir(folder))
            for filename in listOfFiles:
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath) and filename.startswith("wrf"):
                    wrfData = WRFData(filepath, variables=self.variables)
                    wrfFile = wrfData.getData()
                    self.wrfFiles.append(wrfFile)
        else:
            wrfData = WRFData(wrfFilenameOrFilesPath, variables=self.variables)
            wrfFile = wrfData.getData()
            self.wrfFiles.append(wrfFile)
        
    def make(self):
        try:
            os.remove(self.dst)
        except:
            pass

        t = 0
        
        for wrfFile in self.wrfFiles:
            print(wrfFile["path"])

            datetimeStr = wrfFile["datetimeStr"].split("_")
            dateStr=str("".join(datetimeStr[0].split("-"))).replace("b'","")
            timeStr=str("".join(datetimeStr[1].split(":"))).replace("'","")

            LONXY, LATXY = np.meshgrid(self.ww3Grid.lons, self.ww3Grid.lats)
            bilinearInterpolatorRho = Interpolator(wrfFile["XLAT"], wrfFile["XLONG"], LATXY, LONXY)
            
            u10m = bilinearInterpolatorRho.simpleInterp(wrfFile["U10M"], 1e37)
            v10m = bilinearInterpolatorRho.simpleInterp(wrfFile["V10M"], 1e37)

            f = open(self.dst, "a+")
            f.write(dateStr + " " + timeStr + "\n")
            for j in range(0, self.ww3Grid.nrows):
                line = ""
                for i in range(0, self.ww3Grid.ncols):
                    line = "%s%5.2f " % (line, u10m[j][i])
                line = line.strip()
                f.write(line + "\n")
            
            for j in range(0, self.ww3Grid.nrows):
                line = ""
                for i in range(0, self.ww3Grid.ncols):
                    line = "%s%5.2f " % (line, v10m[j][i])
                line = line.strip()
                f.write(line + "\n")

            if t==13:
                maxValue = -1
                minValue = 999
                values = np.zeros(shape=(self.ww3Grid.nrows, self.ww3Grid.ncols))

                for j in range(0, self.ww3Grid.nrows):
                    for i in range(0, self.ww3Grid.ncols):
                        value = math.sqrt((u10m[j][i] ** 2) + (v10m[j][i] ** 2))
                        if value > maxValue:
                            maxValue = value
                        if value < minValue:
                            minValue = value
                        values[j,i] = value
                    
                f1 = open(self.dst + ".grd", "w")
                f1.write("DSAA\n") 
                f1.write(str(self.ww3Grid.ncols) + " " + str(self.ww3Grid.nrows) + "\n") 
                f1.write(str(self.ww3Grid.minLon) + " " + str(self.ww3Grid.maxLon) + "\n") 
                f1.write(str(self.ww3Grid.minLat) + " " + str(self.ww3Grid.maxLat) + "\n")
                f1.write("%5.2f %5.2f\n" % (minValue, maxValue))
                for j in range(0, self.ww3Grid.nrows):
                    line = ""
                    for i in range(0, self.ww3Grid.ncols):
                        line = "%s%5.2f " % (line, values[j][i])
                    line = line.strip()
                    f1.write(line + "\n")

            f.close()
            t = t+1
