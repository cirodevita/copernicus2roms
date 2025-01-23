import os
import time

import ephem
import numpy as np
import argparse

from jncregridder.data.copernicus.Copernicus import CopernicusTem, CopernicusSal, CopernicusSSH, CopernicusCur
from jncregridder.roms.ROMSBoundary import ROMSBoundary
from jncregridder.roms.ROMSGrid import ROMSGrid
from jncregridder.roms.ROMSInit import ROMSInit
from jncregridder.util.Interpolator import BilinearInterpolator, BilinearInterpolator3D


def calculate_julian_date(year, month, day):
    date = ephem.Date((year, month, day))
    julian_date = ephem.julian_date(date)

    return julian_date


def getModSimStartDate(ncepDate):
    year = int(ncepDate[:4])
    month = int(ncepDate[4:6])
    day = int(ncepDate[6:8])

    dSimStartDate = calculate_julian_date(year, month, day)
    dModOffset = calculate_julian_date(1968, 5, 23)

    dModSimStartDate = dSimStartDate - dModOffset

    return dModSimStartDate


class Copernicus2ROMS:
    def __init__(self, gridPath, dataPath, ncepDate, initPath, boundaryPath, vtransform, vstretching, theta_b, theta_s, Tcline, N):
        # Path to the ROMS grid
        self.romsGridPath = gridPath

        # Path to the copernicus current copernicus-data
        self.copernicusPathCur = os.path.join(dataPath, f"myoc_d00_{ncepDate}_cur.nc")

        # Path to the copernicus temperature copernicus-data
        self.copernicusPathTem = os.path.join(dataPath, f"myoc_d00_{ncepDate}_tem.nc")

        # Path to the copernicus salinity copernicus-data
        self.copernicusPathSal = os.path.join(dataPath, f"myoc_d00_{ncepDate}_sal.nc")

        # Path to the copernicus sea surface height copernicus-data
        self.copernicusPathSSH = os.path.join(dataPath, f"myoc_d00_{ncepDate}_ssh.nc")

        # Path to the output init file
        self.romsInitPath = initPath

        # Path to the output boundary file
        self.romsBoundaryPath = boundaryPath

        # Open ROMS grid copernicus-data
        romsGrid = ROMSGrid(gridPath, vtransform, vstretching, theta_b, theta_s, Tcline, N)

        # Get dimension size
        etaRho = romsGrid.etaRho
        xiRho = romsGrid.xiRho
        etaU = romsGrid.etaU
        xiU = romsGrid.xiU
        etaV = romsGrid.etaV
        xiV = romsGrid.xiV

        print("Rho:\n\teta:", etaRho, "\txi:", xiRho)
        print("U:\n\teta:", etaU, "\txi:", xiU)
        print("V:\n\teta:", etaV, "\txi:", xiV)

        # MASK at rho points
        MASKRHO = romsGrid.MASKRHO

        # MASK at u points
        MASKU = romsGrid.MASKU

        # MASK at v points
        MASKV = romsGrid.MASKV

        # LAT,LON at rho points
        LATRHO = romsGrid.LATRHO
        LONRHO = romsGrid.LONRHO

        # Z at rho/sigma points
        romsZ = romsGrid.Z

        # LAT,LON at u points
        LATU = romsGrid.LATU
        LONU = romsGrid.LONU

        # LAT,LON at v points
        LATV = romsGrid.LATV
        LONV = romsGrid.LONV

        # Get the angle between the xi axis and the real east
        ANGLE = romsGrid.ANGLE

        print("MASKRHO:", MASKRHO.shape)
        print("MASKU:", MASKU.shape)
        print("MASKV:", MASKV.shape)
        print("LATRHO:", LATRHO.shape)
        print("LONRHO:", LONRHO.shape)
        print("LATU:", LATU.shape)
        print("LONU:", LONU.shape)
        print("LATV:", LATV.shape)
        print("LONV:", LONV.shape)
        print("ANGLE:", ANGLE.shape)
        print("Z:", romsZ.shape)

        dataTem = CopernicusTem(self.copernicusPathTem)
        dataSal = CopernicusSal(self.copernicusPathSal)
        dataSSH = CopernicusSSH(self.copernicusPathSSH)
        dataCur = CopernicusCur(self.copernicusPathCur)

        # LON at XY points
        # LATXY = dataCur.LAT
        # LONXY = dataCur.LON
        LONXY, LATXY = np.meshgrid(dataCur.LON, dataCur.LAT)
        LATXY = LATXY.filled(np.nan)
        LONXY = LONXY.filled(np.nan)
        copernicusZ = dataCur.Z

        # Set the number of forcing time steps
        forcingTimeSteps = len(dataCur.TIME)

        oceanTime = []
        scrumTime = []
        for t in range(forcingTimeSteps):
            # Set the value of each ocean time as delta form the simulation starting date
            oceanTime.append((getModSimStartDate(ncepDate) + t))
            # Set the scrum time
            scrumTime.append(t * 86400)

        # Instantiate a ROMS init file
        romsInit = ROMSInit(self.romsInitPath, romsGrid, forcingTimeSteps)
        romsInit.OCEAN_TIME = oceanTime
        romsInit.SCRUM_TIME = scrumTime
        romsInit.make()

        # Instantiate a ROMS boundary file
        romsBoundary = ROMSBoundary(self.romsBoundaryPath, romsGrid, forcingTimeSteps)
        romsBoundary.OCEAN_TIME = oceanTime
        romsBoundary.make()

        for t in range(forcingTimeSteps):
            print(f"Time: {t} {oceanTime[t]}")

            # 2D sea surface height
            valuesSSH = dataSSH.ZOS[t]

            # 3D current U component
            valuesU = dataCur.UO[t]

            # 3D current V component
            valuesV = dataCur.VO[t]

            # 3D temperature
            valuesTem = dataTem.THETAO[t]

            # 3D salinity
            valuesSal = dataSal.SO[t]

            # Create a 2D bilinear interpolator on Rho points
            bilinearInterpolatorRho = BilinearInterpolator(LATXY, LONXY, LATRHO, LONRHO, MASKRHO)

            # Interpolate the SSH
            print("Interpolating SSH")
            SSH_ROMS = bilinearInterpolatorRho.interp(valuesSSH, dataSSH.FillValue)

            # Create a 3D bilinear interpolator on Rho points
            interpolator3DRho = BilinearInterpolator3D(LATXY, LONXY, copernicusZ, LATRHO, LONRHO, romsZ, MASKRHO, romsGrid)
            # Create a 3D bilinear interpolator on U points
            interpolator3DU = BilinearInterpolator3D(LATXY, LONXY, copernicusZ, LATU, LONU, romsZ, MASKU, romsGrid)
            # Create a 3D bilinear interpolator on V points
            interpolator3DV = BilinearInterpolator3D(LATXY, LONXY, copernicusZ, LATV, LONV, romsZ, MASKV, romsGrid)

            print("Interpolating SAL")
            SAL_ROMS = interpolator3DRho.interp(valuesSal, dataSal.FillValue)

            print("Interpolating TEMP")
            TEM_ROMS = interpolator3DRho.interp(valuesTem, dataTem.FillValue)

            print("Interpolating V")
            V_ROMS = interpolator3DV.interp(valuesV, dataCur.FillValue)

            print("Interpolating U")
            U_ROMS = interpolator3DU.interp(valuesU, dataCur.FillValue)

            print("Calculating UBAR")
            UBAR = np.ma.mean(U_ROMS, axis=0)
            UBAR = np.ma.masked_where(MASKU != 1, UBAR)

            print("Calculating VBAR")
            VBAR = np.mean(V_ROMS, axis=0)
            VBAR = np.ma.masked_where(MASKV != 1, VBAR)

            print(f"Time: {t} Saving init file...")
            romsInit.ZETA = SSH_ROMS
            romsInit.SALT = SAL_ROMS
            romsInit.TEMP = TEM_ROMS
            romsInit.U = U_ROMS
            romsInit.V = V_ROMS
            romsInit.UBAR = UBAR
            romsInit.VBAR = VBAR
            romsInit.write(t)

            print(f"Time: {t} Saving bry file...")
            romsBoundary.ZETA = SSH_ROMS
            romsBoundary.SALT = SAL_ROMS
            romsBoundary.TEMP = TEM_ROMS
            romsBoundary.U = U_ROMS
            romsBoundary.V = V_ROMS
            romsBoundary.UBAR = UBAR
            romsBoundary.VBAR = VBAR
            romsBoundary.write(t)

        romsInit.close()
        romsBoundary.close()


def parser():
    parser = argparse.ArgumentParser(description="Copernicus2ROMS")
    parser.add_argument("--gridPath", type=str, required=True, help="Path to grid copernicus-data file")
    parser.add_argument("--dataPath", type=str, required=True, help="Path to general copernicus-data directory")
    parser.add_argument("--ncepDate", type=str, required=True,help="Date for NCEP copernicus-data")
    parser.add_argument("--initPath", type=str, required=True, help="Path to store initial copernicus-data file")
    parser.add_argument("--boundaryPath", type=str, required=True, help="Path to store boundary copernicus-data file")
    parser.add_argument("--vtransform", type=int, default=2, help="Vertical transformation parameter (default: 2)")
    parser.add_argument("--vstretching", type=int, default=4, help="Vertical stretching parameter (default: 4)")
    parser.add_argument("--theta_b", type=float, default=0.4, help="Bottom slope parameter (default: 0.4)")
    parser.add_argument("--theta_s", type=float, default=3.0, help="Surface slope parameter (default: 3.0)")
    parser.add_argument("--Tcline", type=float, default=25, help="Critical depth (default: 25)")
    parser.add_argument("--N", type=int, default=30, help="Number of vertical levels (default: 30)")
    return parser


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()

    start = time.time()
    Copernicus2ROMS(args.gridPath, args.dataPath, args.ncepDate, args.initPath, args.boundaryPath, args.vtransform, args.vstretching, args.theta_b, args.theta_s, args.Tcline, args.N)
    end = time.time() - start

    print(f"Execution time: {end}\n")


if __name__ == '__main__':
    main()
