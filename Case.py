import numpy as np
from ANSYSfluent import readCase as rc


class Case:
    def __init__(self,filename) -> None:
        self.nodes,self.faces,self.cells,self.faceCenters,self.cellCenters,self.c0,self.c1,self.zoneId,self.zonetype,self.minId,self.maxId = rc(filename,returnMore=True)
    
