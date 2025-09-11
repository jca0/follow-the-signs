from pprint import pprint
import numpy as np
from .img_utils import *
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # sim_env/baselines
POLYCAM_DIR = os.path.join(BASE_DIR, "polycam_scans")

class Bldg4:
    def __init__(self):
        self.occupancy_grid = img_to_occgrid(os.path.join(POLYCAM_DIR, "bldg4.png"), resolution=10)
        row, col = self.occupancy_grid.shape
        
        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}
        
        self.semantic_grid[88][15]["room_number"] = "101"
        self.semantic_grid[88][20]["room_number"] = "104"
        self.semantic_grid[74][15]["room_number"] = "103"
        self.semantic_grid[70][20]["room_number"] = "108"
        self.semantic_grid[49][20]["room_number"] = "110"
        self.semantic_grid[60][15]["room_number"] = "107"
        self.semantic_grid[50][15]["room_number"] = "109"
        self.semantic_grid[36][15]["room_number"] = "113"
        self.semantic_grid[23][15]["room_number"] = "115"
        self.semantic_grid[14][7]["room_number"] = "119"
        self.semantic_grid[8][15]["room_number"] = "119"
        self.semantic_grid[18][31]["room_number"] = "129"
        self.semantic_grid[18][48]["room_number"] = "131"
        self.semantic_grid[18][73]["room_number"] = "131"
        self.semantic_grid[18][91]["room_number"] = "141"
        self.semantic_grid[18][100]["room_number"] = "143"
        self.semantic_grid[18][114]["room_number"] = "145"
        self.semantic_grid[18][132]["room_number"] = "149"
        self.semantic_grid[18][151]["room_number"] = "153"
        self.semantic_grid[18][168]["room_number"] = "159"
        self.semantic_grid[18][181]["room_number"] = "161"
        self.semantic_grid[18][195]["room_number"] = "163"
        self.semantic_grid[18][208]["room_number"] = "167"
        self.semantic_grid[18][213]["room_number"] = "169"
        self.semantic_grid[18][217]["room_number"] = "173" 
        self.semantic_grid[28][235]["room_number"] = "172"
        self.semantic_grid[23][48]["room_number"] = "132"
        self.semantic_grid[23][62]["room_number"] = "138"
        self.semantic_grid[23][71]["room_number"] = "140"
        self.semantic_grid[22][80]["room_number"] = "142"
        self.semantic_grid[22][90]["room_number"] = "144"
        self.semantic_grid[22][100]["room_number"] = "146"
        self.semantic_grid[22][110]["room_number"] = "148"
        self.semantic_grid[22][122]["room_number"] = "152"
        self.semantic_grid[22][142]["room_number"] = "156"
        self.semantic_grid[22][160]["room_number"] = "158"
        self.semantic_grid[22][180]["room_number"] = "162"
        self.semantic_grid[22][194]["room_number"] = "164"
        self.semantic_grid[22][201]["room_number"] = "166"

        self.semantic_grid[88][15]["sign"] = "Rooms 101-174 ahead."
        self.semantic_grid[56][15]["sign"] = "Rooms 101-108 southward. Rooms 109-174 northward."
        self.semantic_grid[36][30]["sign"] = "Rooms 113-174 ahead. Rooms 101-110 southward."
        self.semantic_grid[14][15]["sign"] = "Rooms 101-115 southward. Rooms 129-174 to the right."
        self.semantic_grid[29][38]["sign"] = "Rooms 101-129 to the left. Rooms 131-174 to the right."
        self.semantic_grid[22][131]["sign"] = "Rooms 101-152 to the left. Rooms 156-174 to the right."
        self.semantic_grid[18][178]["sign"] = "Rooms 101-159 to the left. Rooms 161-174 to the right."


class LargeMaseeh:
    def __init__(self):
        self.occupancy_grid = img_to_occgrid(os.path.join(POLYCAM_DIR, "maseeh.png"), resolution=10)
        row, col = self.occupancy_grid.shape

        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}

        self.semantic_grid[124][16]["room_number"] = "6001"
        self.semantic_grid[124][20]["room_number"] = "6002"
        self.semantic_grid[121][15]["room_number"] = "6003"
        self.semantic_grid[116][15]["room_number"] = "6005"
        self.semantic_grid[111][15]["room_number"] = "6007"
        self.semantic_grid[104][15]["room_number"] = "6009"
        self.semantic_grid[93][15]["room_number"] = "6011"
        self.semantic_grid[85][15]["room_number"] = "6013"
        self.semantic_grid[76][15]["room_number"] = "6015"
        self.semantic_grid[69][15]["room_number"] = "6017"
        self.semantic_grid[53][15]["room_number"] = "6019"
        self.semantic_grid[38][15]["room_number"] = "6027"
        self.semantic_grid[28][15]["room_number"] = "6029"
        self.semantic_grid[18][15]["room_number"] = "6033"
        self.semantic_grid[121][21]["room_number"] = "6004"
        self.semantic_grid[116][21]["room_number"] = "6006"
        self.semantic_grid[111][21]["room_number"] = "6008"
        self.semantic_grid[104][21]["room_number"] = "6010"
        self.semantic_grid[93][21]["room_number"] = "6012"
        self.semantic_grid[85][20]["room_number"] = "6014"
        self.semantic_grid[76][20]["room_number"] = "6016"
        self.semantic_grid[69][20]["room_number"] = "6018"
        self.semantic_grid[61][20]["room_number"] = "6020"
        self.semantic_grid[53][20]["room_number"] = "6022"
        self.semantic_grid[40][20]["room_number"] = "6040"
        self.semantic_grid[28][20]["room_number"] = "6032"
        self.semantic_grid[18][20]["room_number"] = "6034"
        self.semantic_grid[11][10]["room_number"] = "6035"
        self.semantic_grid[10][12]["room_number"] = "6037"
        self.semantic_grid[10][23]["room_number"] = "6038"
        self.semantic_grid[11][25]["room_number"] = "6036"
        self.semantic_grid[50][24]["room_number"] = "6022"
        self.semantic_grid[50][32]["room_number"] = "6041"
        self.semantic_grid[50][51]["room_number"] = "6047"
        self.semantic_grid[50][77]["room_number"] = "6051"
        self.semantic_grid[50][92]["room_number"] = "6053"
        self.semantic_grid[46][26]["room_number"] = "6040"
        self.semantic_grid[46][43]["room_number"] = "6044"
        self.semantic_grid[46][55]["room_number"] = "6046"
        self.semantic_grid[46][65]["room_number"] = "6048"
        self.semantic_grid[46][77]["room_number"] = "6052"
        self.semantic_grid[46][92]["room_number"] = "6054"
        self.semantic_grid[40][101]["room_number"] = "6054"
        self.semantic_grid[28][101]["room_number"] = "6090"
        self.semantic_grid[18][101]["room_number"] = "6096"
        self.semantic_grid[12][96]["room_number"] = "6098"
        self.semantic_grid[10][97]["room_number"] = "6100"
        self.semantic_grid[10][102]["room_number"] = "6101"
        self.semantic_grid[10][107]["room_number"] = "6099"
        self.semantic_grid[12][109]["room_number"] = "6097"
        self.semantic_grid[20][105]["room_number"] = "6093"
        self.semantic_grid[28][105]["room_number"] = "6091"
        self.semantic_grid[37][105]["room_number"] = "6087"
        self.semantic_grid[53][105]["room_number"] = "6083"
        self.semantic_grid[80][105]["room_number"] = "6081"
        self.semantic_grid[62][100]["room_number"] = "6078"
        self.semantic_grid[80][100]["room_number"] = "6076"
        self.semantic_grid[98][103]["room_number"] = "6065"

        self.semantic_grid[66][15]["sign"] = "Rooms 6001-6018 southward. Rooms 6019-6041 northward."
        self.semantic_grid[35][15]["sign"] = "Rooms 6029-6038 northward. Rooms 6001-6027 southward. Rooms 6050-6101 to the right."
        self.semantic_grid[50][28]["sign"] = "Rooms 6001-6040 to the left. Rooms 6041-6101 to the right."
        self.semantic_grid[46][60]["sign"] = "Rooms 6001-6047 to the left. Rooms 6048-6101 to the right."
        self.semantic_grid[44][105]["sign"] = "Rooms 6087-6101 northward. Rooms 6053-6083 southward."
        self.semantic_grid[66][100]["sign"] = "Rooms 6065-6081 southward. Rooms 6083-6101 northward."


class LargeSchwarz:
    def __init__(self):
        self.occupancy_grid = img_to_occgrid(os.path.join(POLYCAM_DIR, "schwarzman.png"), resolution=10)
        row, col = self.occupancy_grid.shape

        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}


class NoisySchwarz:
    def __init__(self):
        self.occupancy_grid = img_to_occgrid(os.path.join(POLYCAM_DIR, "noisy_schwarzman.png"), resolution=10)
        row, col = self.occupancy_grid.shape

        self.semantic_grid = [[{} for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if self.occupancy_grid[y][x] == 1:
                    self.semantic_grid[y][x] = {"feature_type": "wall"}
                else:
                    self.semantic_grid[y][x] = {"feature_type": "free"}