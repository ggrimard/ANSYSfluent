import numpy as np

class Case:
    def __init__(self,filename) -> None:
        from ANSYSfluent import readCase as rc
        from ANSYSfluent import getZones as gz
        self.zones = gz(filename)
        if len(self.zones)==1:
            self.nodes,self.faces,self.cells,self.faceCenters,self.cellCenters,self.c0,self.c1,self.zoneId,self.zonetype,self.minId,self.maxId = rc(filename,zone=int(self.zones),returnMore=True)
        else:
            print(f'case has multiple zones:\n {self.zones}')
            for zid in self.zones:
                exec("self.z%s = Zone(filename,int(%s))"%(zid,zid))



class Zone:
    def __init__(self,filename,ZONEID) -> None:
        from ANSYSfluent import readCase as rc
        self.nodes,self.faces,self.cells,self.faceCenters,self.cellCenters,self.c0,self.c1,self.zoneId,self.zonetype,self.minId,self.maxId = rc(filename,ZONEID,returnMore=True)