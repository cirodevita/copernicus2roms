import argparse
import time

from jncregridder.wwatch3.WW3Wind import WW3Wind
from jncregridder.wwatch3.WW3Grid import WW3Grid


class WRF2WW3:
    def __init__(self, wrfFilename, ww3GridFilename, forcingFilename):
        self.ww3GridFilename = ww3GridFilename
        self.wrfFilenameOrFilesPath = wrfFilename
        self.ww3ForcingFilename = forcingFilename

        # Open the WW3 Grid
        ww3Grid = WW3Grid(self.ww3GridFilename)

        # Open WW3 wind File
        ww3Wind = WW3Wind(self.ww3ForcingFilename, ww3Grid, self.wrfFilenameOrFilesPath)

        start = time.time()
        ww3Wind.make()
        end = time.time()
        print(end-start)

def parser():
    parser = argparse.ArgumentParser(description="WRF2WW3")
    parser.add_argument("--ww3GridFilename", type=str, required=True, help="Path to grid ww3-data file")
    parser.add_argument("--wrfFilename", type=str, required=True, help="Path to wrf file/files")
    parser.add_argument("--forcingFilename", type=str, required=True, help="")
    return parser


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()

    WRF2WW3(args.wrfFilename, args.ww3GridFilename, args.forcingFilename)


if __name__ == '__main__':
    main()