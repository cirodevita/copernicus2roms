from netCDF4 import Dataset
import numpy as np
import jncregridder.roms.vgrid as vgrid


class ROMSGrid:
    def __init__(self, url, vtransform=2, vstretching=4, theta_b=0.4, theta_s=3.0, Tcline=25, N=30):
        self.url = url
        self.ncDataset = Dataset(url)
        self.no_data = 1e37
        self.zeta = 0
        self.vtransform = vtransform
        self.vstretching = vstretching
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.TCLINE = Tcline
        self.N = N

        # Find dimensions
        self.dimEtaRho = self.ncDataset.dimensions["eta_rho"]
        self.dimXiRho = self.ncDataset.dimensions["xi_rho"]
        self.dimEtaPsi = self.ncDataset.dimensions["eta_psi"]
        self.dimXiPsi = self.ncDataset.dimensions["xi_psi"]
        self.dimEtaU = self.ncDataset.dimensions["eta_u"]
        self.dimXiU = self.ncDataset.dimensions["xi_u"]
        self.dimEtaV = self.ncDataset.dimensions["eta_v"]
        self.dimXiV = self.ncDataset.dimensions["xi_v"]

        # Get dimension lengths
        self.etaRho = len(self.dimEtaRho)
        self.xiRho = len(self.dimXiRho)
        self.etaPsi = len(self.dimEtaPsi)
        self.xiPsi = len(self.dimXiPsi)
        self.etaU = len(self.dimEtaU)
        self.xiU = len(self.dimXiU)
        self.etaV = len(self.dimEtaV)
        self.xiV = len(self.dimXiV)

        self.H = self.__load("h")
        self.zeta = np.zeros_like(self.H)

        if vtransform == 1 and vstretching == 1:
            self.s_coord = vgrid.s_coordinate(self.H, self.theta_b, self.theta_s, self.TCLINE, self.N, zeta=self.zeta)
        elif vtransform == 2 and vstretching == 2:
            self.s_coord = vgrid.s_coordinate_2(self.H, self.theta_b, self.theta_s, self.TCLINE, self.N, zeta=self.zeta)
        elif vtransform == 2 and vstretching == 4:
            self.s_coord = vgrid.s_coordinate_4(self.H, self.theta_b, self.theta_s, self.TCLINE, self.N, zeta=self.zeta)
        elif vtransform == 2 and vstretching == 5:
            self.s_coord = vgrid.s_coordinate_5(self.H, self.theta_b, self.theta_s, self.TCLINE, self.N, zeta=self.zeta)
        else:
            raise ValueError(f"Unsupported Vtransform={vtransform} and Vstretching={vstretching}")

        self.HC = self.s_coord.hc
        self.s_rho = self.s_coord.s_rho
        self.cs_r = self.s_coord.Cs_r
        # self.s_w = self.s_coord.s_w
        # self.cs_w = self.s_coord.Cs_w
        self.Z = self.s_coord.z_r[:]

        self.ANGLE = self.__load('angle')
        self.LATRHO = self.__load('lat_rho')
        self.LONRHO = self.__load('lon_rho')
        self.LATPSI = self.__load('lat_psi')
        self.LONPSI = self.__load('lon_psi')
        self.LATU = self.__load('lat_u')
        self.LONU = self.__load('lon_u')
        self.LATV = self.__load('lat_v')
        self.LONV = self.__load('lon_v')
        self.MASKRHO = self.__load('mask_rho')
        self.MASKU = self.__load('mask_u')
        self.MASKV = self.__load('mask_v')

    def __load(self, variable_name):
        try:
            return self.ncDataset.variables[variable_name][:]
        except KeyError:
            raise Exception(f"Variable {variable_name} not found in the dataset.")
        except Exception as e:
            raise Exception(f"Error loading variable {variable_name}: {str(e)}")
