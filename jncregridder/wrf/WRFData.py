from netCDF4 import Dataset
from wrf import getvar
import numpy as np
import jdcal


class WRFData:
    def __init__(self, url, variables=None):
        self.url = url

        self.simStartDate = None
        self.dModDate = None

        self.ncDataset = Dataset(url)
        self.dimTime = self.ncDataset.dimensions.get("Time")
        self.timeDim = len(self.dimTime)

        self.datetimeStr = ''.join([x.decode() for x in self.ncDataset.variables["Times"][:][0]])
        year = int(self.datetimeStr[0:4])
        month = int(self.datetimeStr[5:7])
        day = int(self.datetimeStr[8:10])
        hour = int(self.datetimeStr[11:13])
        minute = int(self.datetimeStr[14:16])
        second = int(self.datetimeStr[17:19])

        dDate = sum(jdcal.gcal2jd(year, month, day)) + hour / 24.0 + minute / 1440.0 + second / 86400.0
        dModOffset = sum(jdcal.gcal2jd(1968, 5, 23))
        self.dModDate = dDate - dModOffset

        self.variables = variables if variables else ["XLAT", "XLONG", "T2", "SLP", "uvmet10", "rh2", "cloudfrac", "SWDOWN", "GLW"]

        self.XLAT = None
        self.XLONG = None
        self.U10M = None
        self.V10M = None
        self.T2 = None
        self.SLP = None
        self.RH2 = None
        self.CLF = None
        self.SWDOWN = None
        self.GLW = None

        self._load_variables()

    def _load_variables(self):
        if "XLAT" in self.variables:
            self.XLAT = np.array(self.__loadWrf("XLAT"))
        if "XLONG" in self.variables:
            self.XLONG = np.array(self.__loadWrf("XLONG"))
        if "T2" in self.variables:
            self.T2 = np.array(self.__load("T2") - 273.15)
        if "SLP" in self.variables:
            self.SLP = np.array(self.__loadWrf("slp"))
        if "uvmet10" in self.variables:
            self.UVMET10 = np.array(self.__loadWrf("uvmet10"))
            self.U10M = np.array(self.UVMET10[0])
            self.V10M = np.array(self.UVMET10[1])
        if "rh2" in self.variables:
            self.RH2 = np.array(self.__loadWrf("rh2"))
        if "cloudfrac" in self.variables:
            self.CLFR = np.array(self.__loadWrf("cloudfrac"))
            self.CLF = np.maximum(self.CLFR[0], self.CLFR[1])
        if "SWDOWN" in self.variables:
            self.SWDOWN = np.array(self.__load("SWDOWN"))
        if "GLW" in self.variables:
            self.GLW = np.array(self.__load("GLW"))

    def __loadWrf(self, variable_name):
        try:
            return getvar(self.ncDataset, variable_name, meta=False)
        except KeyError:
            raise Exception(f"Variable {variable_name} not found in the dataset.")
        except Exception as e:
            raise Exception(f"Error loading variable {variable_name}: {str(e)}")

    def __load(self, variable_name):
        try:
            return self.ncDataset.variables[variable_name][:]
        except KeyError:
            raise Exception(f"Variable {variable_name} not found in the dataset.")
        except Exception as e:
            raise Exception(f"Error loading variable {variable_name}: {str(e)}")

    def getData(self):
        data = {"path": self.url, "dModDate": self.dModDate, "datetimeStr": self.datetimeStr}

        if "XLAT" in self.variables:
            data["XLAT"] = self.XLAT
        if "XLONG" in self.variables:
            data["XLONG"] = self.XLONG
        if "T2" in self.variables:
            data["T2"] = self.T2[:]
        if "SLP" in self.variables:
            data["SLP"] = self.SLP[:]
        if "uvmet10" in self.variables:
            data["U10M"] = self.U10M
            data["V10M"] = self.V10M
        if "rh2" in self.variables:
            data["RH2"] = self.RH2[:]
        if "cloudfrac" in self.variables:
            data["CLF"] = self.CLF
        if "SWDOWN" in self.variables:
            data["SWDOWN"] = self.SWDOWN[:]
        if "GLW" in self.variables:
            data["GLW"] = self.GLW[:]

        return data
