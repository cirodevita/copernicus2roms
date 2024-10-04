from scipy.interpolate import griddata, interp1d
from concurrent.futures import ProcessPoolExecutor
from numpy.ma.core import MaskedArray
from numba import jit
import numpy as np
import multiprocessing


def gridInterp(srcLAT, srcLON, values, dstLAT, dstLON, fillValue, method):
    if isinstance(values, MaskedArray):
        values = values.filled(fill_value=np.nan)

    return griddata(
        (srcLAT.flatten(), srcLON.flatten()),
        values.flatten(),
        (dstLAT, dstLON),
        fill_value=fillValue,
        method=method
    )


@jit(nopython=True)
def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


@jit(nopython=True)
def interp(src, srcMissingValue, dstMissingValue, dstLAT, dstLON, srcLAT, srcLON, dstMASK):
    USE_IDW = True

    dstEta = len(dstLAT)
    dstXi = len(dstLAT[0])
    srcEta = len(srcLAT)
    srcXi = len(srcLAT[0])

    srcLonMin = srcLON[0][0]
    srcLatMin = srcLAT[0][0]
    srcLonMax = srcLON[srcEta - 1][srcXi - 1]
    srcLatMax = srcLAT[srcEta - 1][srcXi - 1]
    srcLatDelta = srcLatMax - srcLatMin
    srcLonDelta = srcLonMax - srcLonMin
    srcLatStep = srcLatDelta / srcEta
    srcLonStep = srcLonDelta / srcXi

    dst = np.full((dstEta, dstXi), dstMissingValue)

    for dstJ in range(dstEta):
        for dstI in range(dstXi):
            if dstMASK[dstJ][dstI] == 1:
                dstLon = dstLON[dstJ][dstI]
                dstLat = dstLAT[dstJ][dstI]

                srcII = (dstLon - srcLonMin) / srcLonStep
                srcJJ = (dstLat - srcLatMin) / srcLatStep

                iR = int(srcII)
                jR = int(srcJJ)

                pointsBilinear = []

                for j in [jR - 1, jR + 1]:
                    for i in [iR - 1, iR + 1]:
                        jj = min(max(j, 0), srcEta - 1)
                        ii = min(max(i, 0), srcXi - 1)
                        if not np.isnan(src[jj][ii]) and src[jj][ii] != srcMissingValue:
                            pointsBilinear.append((jj, ii, src[jj][ii]))

                if len(pointsBilinear) == 4:
                    lon1 = srcLON[pointsBilinear[0][0]][pointsBilinear[0][1]]
                    lat1 = srcLAT[pointsBilinear[0][0]][pointsBilinear[0][1]]
                    lon2 = srcLON[pointsBilinear[1][0]][pointsBilinear[1][1]]
                    lat2 = srcLAT[pointsBilinear[2][0]][pointsBilinear[2][1]]

                    dLon = lon2 - lon1
                    dLat = lat2 - lat1

                    FXY1 = ((lon2 - dstLon) / dLon) * pointsBilinear[0][2] + ((dstLon - lon1) / dLon) * \
                           pointsBilinear[1][2]
                    FXY2 = ((lon2 - dstLon) / dLon) * pointsBilinear[2][2] + ((dstLon - lon1) / dLon) * \
                           pointsBilinear[3][2]

                    dst[dstJ][dstI] = ((lat2 - dstLat) / dLat) * FXY1 + ((dstLat - lat1) / dLat) * FXY2

                elif USE_IDW:
                    pointsIDW = []
                    size = 0

                    while len(pointsIDW) == 0 and size < 4:
                        size += 1
                        for j in range(jR - size, jR + size + 1):
                            jj = min(max(j, 0), srcEta - 1)
                            for i in range(iR - size, iR + size + 1):
                                ii = min(max(i, 0), srcXi - 1)
                                if not np.isnan(src[jj][ii]) and src[jj][ii] != srcMissingValue:
                                    pointsIDW.append(
                                        (jj, ii, haversine(
                                            dstLat, dstLon,
                                            srcLAT[jj][ii], srcLON[jj][ii]
                                        ), src[jj][ii])
                                    )

                    if len(pointsIDW) > 0:
                        weighted_values_sum = 0.0
                        sum_of_weights = 0.0
                        for point in pointsIDW:
                            weight = 1 / point[2]
                            # weight = 1 / (point[2] ** 2)
                            sum_of_weights += weight
                            weighted_values_sum += weight * point[3]
                        dst[dstJ][dstI] = weighted_values_sum / sum_of_weights
    return dst


def interp_horizontal(k, srcLAT, srcLON, srcZ, values, dstLAT, dstLON, dstMask, fillValue):
    print(f"<k={k} depth:{srcZ[k][0][0]:.2f}>")

    result = interp(values[k].filled(np.nan), fillValue, np.nan, dstLAT, dstLON, srcLAT, srcLON, dstMask)
    return result


@jit(nopython=True)
def vertical_interp(dstLevs, srcEta, srcXi, tSrc, srcZ, dstZ, dstMask):
    tDst = np.full((dstLevs, srcEta, srcXi), 1e37)
    mask_indices = np.where(dstMask == 1)

    for i, j in zip(*mask_indices):
        vertical_profile = tSrc[:, i, j]
        first_nan_index = -1
        if np.isnan(vertical_profile).any():
            first_nan_index = np.argmax(np.isnan(vertical_profile))

        vertical_profile = tSrc[:first_nan_index, i, j][::-1]
        z_levels = -np.array(srcZ[:first_nan_index])[::-1]

        target_z_levels = dstZ[:, i, j]
        tDst[:, i, j] = np.interp(target_z_levels, z_levels, vertical_profile)

    return tDst


class Interpolator:
    def __init__(self, srcLAT, srcLON, dstLAT, dstLON, dstMASK=None, method="linear"):
        self.srcLAT = srcLAT
        self.srcLON = srcLON
        if isinstance(self.srcLAT, MaskedArray):
            self.srcLAT = srcLAT.filled(np.nan)
        if isinstance(self.srcLON, MaskedArray):
            self.srcLON = srcLON.filled(np.nan)

        self.dstLAT = dstLAT
        self.dstLON = dstLON
        if isinstance(self.dstLAT, MaskedArray):
            self.dstLAT = dstLAT.filled(np.nan)
        if isinstance(self.dstLON, MaskedArray):
            self.dstLON = dstLON.filled(np.nan)

        if dstMASK is not None:
            self.dstMASK = dstMASK.filled(np.nan)
        
        self.method = method

        self.dstSNDim = len(dstLAT)
        self.dstWEDim = len(dstLAT[0])

    def interp(self, values, fillValue):
        interp_values = interp(values.filled(np.nan), fillValue, 1e37, self.dstLAT, self.dstLON, self.srcLAT,
                               self.srcLON, self.dstMASK)
        return interp_values

    def simpleInterp(self, values, fillValue):
        interp_values = gridInterp(self.srcLAT, self.srcLON, values, self.dstLAT, self.dstLON, fillValue, self.method)
        return interp_values


class BilinearInterpolator(Interpolator):
    def __init__(self, srcLAT, srcLON, dstLAT, dstLON, dstMASK, method="linear"):
        super().__init__(srcLAT, srcLON, dstLAT, dstLON, dstMASK, method)


class BilinearInterpolator3D(Interpolator):
    def __init__(self, srcLAT, srcLON, srcZ, dstLAT, dstLON, dstZ, dstMASK, romsGrid):
        super().__init__(srcLAT, srcLON, dstLAT, dstLON, dstMASK)
        self.srcZ = srcZ.filled(np.nan)
        self.dstZ = dstZ
        self.srcLevs = len(srcZ)
        self.dstLevs = len(romsGrid.s_rho)
        self.romsGrid = romsGrid
        self.srcDepth = []

        self.maxK = 0
        for k in range(self.srcLevs):
            copernicusDepth = self.srcZ[k][0][0]
            self.srcDepth.append(copernicusDepth)

            if copernicusDepth > romsGrid.H.max():
                self.maxK = k + 1
                break

    def interp(self, values, fillValue):
        tSrc = np.empty((self.maxK, self.dstSNDim, self.dstWEDim))

        if isinstance(self.srcZ, MaskedArray):
            self.srcZ = self.srcZ.filled(np.nan)
        if isinstance(self.dstZ, MaskedArray):
            self.dstZ = self.dstZ.filled(np.nan)
        if isinstance(self.dstMASK, MaskedArray):
            self.dstMASK = self.dstMASK.filled(np.nan)

        num_processors = multiprocessing.cpu_count()
        print(f"Number of processes used: {num_processors}")
        with ProcessPoolExecutor(max_workers=num_processors) as executor:
            futures = [executor.submit(interp_horizontal, k, self.srcLAT, self.srcLON, self.srcZ, values,
                                       self.dstLAT, self.dstLON, self.dstMASK, fillValue) for k in range(self.maxK)]

            for k, future in enumerate(futures):
                tSrc[k] = future.result()

        print("Interpolating vertically...")
        tDst = vertical_interp(self.dstLevs, self.dstSNDim, self.dstWEDim, tSrc, self.srcDepth, self.dstZ, self.dstMASK)

        return tDst
