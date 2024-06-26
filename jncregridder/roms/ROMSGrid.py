from netCDF4 import Dataset
import numpy as np


class ROMSGrid:
    def __init__(self, url):
        self.url = url
        self.ncDataset = Dataset(url)
        self.no_data = 1e37
        self.zeta = 0

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

        self.H = self.__load__("h")

        if "hc" not in self.ncDataset.variables:
            self.HC = self.H.min()
            if self.HC > 5:
                self.HC = 5

            self.theta_b = 3.0
            self.theta_s = 0.0
            self.TCLINE = 25
            N = 30

            ds = 1.0 / N
            lev = [(1 + i) - 0.5 for i in range(N)]
            self.s_rho = np.array([(lev[i] - N) * ds for i in range(len(lev))])

            self.s_w = np.empty(N + 1)
            self.s_w[0] = -1
            self.s_w[1:] = self.s_rho + ds / 2

            self.cs_r = np.empty(N)
            self.cs_w = np.empty(N + 1)
            self.cs_w[0] = -1

            if self.theta_s > 0:
                cff1 = 1 / np.sinh(self.theta_s)
                cff2 = 0.5 / np.tanh(0.5 * self.theta_s)

                pTheta = np.sinh(self.theta_s * self.s_rho) * cff1
                rTheta = np.tanh(self.theta_s * (self.s_rho + 0.5)) / (2.0 * np.tanh(0.5 * self.theta_s) - 0.5)
                self.cs_r = (1.0 - self.theta_b) * pTheta + self.theta_b * rTheta

                self.cs_w[1:] = (1.0 - self.theta_b) * cff1 * np.sinh(self.theta_s * self.s_w[1:]) + self.theta_b * (
                            cff2 * np.tanh(self.theta_s * (self.s_w[1:] + 0.5)) - 0.5)
            else:
                self.cs_r = self.s_rho
                self.cs_w[1:] = self.s_w[1:]
        else:
            self.HC = self.__load__('hc')
            self.theta_b = self.__load__('theta_b')
            self.theta_s = self.__load__('theta_s')
            self.TCLINE = self.__load__('Tcline')
            self.s_rho = self.__load__('s_rho')
            self.cs_r = self.__load__('Cs_r')
            self.s_w = self.__load__('s_w')
            self.cs_w = self.__load__('Cs_w')

        self.ANGLE = self.__load__('angle')
        self.LATRHO = self.__load__('lat_rho')
        self.LONRHO = self.__load__('lon_rho')
        self.LATPSI = self.__load__('lat_psi')
        self.LONPSI = self.__load__('lon_psi')
        self.LATU = self.__load__('lat_u')
        self.LONU = self.__load__('lon_u')
        self.LATV = self.__load__('lat_v')
        self.LONV = self.__load__('lon_v')
        self.MASKRHO = self.__load__('mask_rho')
        self.MASKU = self.__load__('mask_u')
        self.MASKV = self.__load__('mask_v')
        self.Z = self.__load__('z')

    def __load__(self, variable_name):
        try:
            if variable_name == "z":
                # Reshape s_rho and cs_r for broadcasting
                s_rho_reshaped = self.s_rho[:, np.newaxis, np.newaxis]
                cs_r_reshaped = self.cs_r[:, np.newaxis, np.newaxis]
                # Calculate S for all k, j, i simultaneously
                S = self.HC * s_rho_reshaped + (self.H - self.HC) * cs_r_reshaped
                # Calculate Z for all k, j, i simultaneously
                Z = S + self.zeta * (1 + (S / self.H))

                variable_data = Z
            else:
                variable_data = self.ncDataset.variables[variable_name][:]

            return variable_data
        except KeyError:
            raise Exception(f"Variable {variable_name} not found in the dataset.")
        except Exception as e:
            raise Exception(f"Error loading variable {variable_name}: {str(e)}")
