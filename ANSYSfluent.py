import numpy as np
import h5py
from numba import njit,prange
from Case import Case

def case(filename):
    return Case(filename)

def getZones(filename):
    with h5py.File(filename, 'r') as f:
        zones = np.array(f['/meshes/'])
    return zones


def readCase(filename,zone=1,returnMore=False):
    """ function that reads a case file and returns the relevant mesh data of the zone"""

    with h5py.File(filename, 'r') as f:
        nr = np.array(f[f'/meshes/{zone}/nodes/coords/'])
        nodes = np.array(f[f'/meshes/{zone}/nodes/coords/{nr[0]}'])
        facenodes = np.array(f[f'/meshes/{zone}/faces/nodes/1/nodes'],dtype=np.int32)-1
        facennodes = np.array(f[f'/meshes/{zone}/faces/nodes/1/nnodes'],dtype=np.int32)
        c0 = np.array(f[f'/meshes/{zone}/faces/c0/1'],dtype=np.int32)-1
        c1 = np.array(f[f'/meshes/{zone}/faces/c1/1'],dtype=np.int32)-1

        zoneId = np.array(f[f'/meshes/{zone}/faces/zoneTopology/id'])
        minId = np.array(f[f'/meshes/{zone}/faces/zoneTopology/minId'])
        maxId = np.array(f[f'/meshes/{zone}/faces/zoneTopology/maxId'])
        zonetype = np.array(f[f'/meshes/{zone}/faces/zoneTopology/zoneType'])

    facennodes = np.insert(facennodes,0,0)
    faces = makeFaceList(facennodes,facenodes)

    cells,c1 = makeCellsList(faces,c0,c1)

    cellCenters = getCellCenters(getCellNodes(cells,faces,nodes),nodes.shape[1])
    faceCenters = getFaceCenters(faces,nodes)

    if returnMore:
        return nodes,faces,cells,faceCenters,cellCenters,c0,c1,zoneId,zonetype,minId,maxId

    return nodes,faces,cells,faceCenters,cellCenters

@njit
def makeFaceList(facennodes,facenodes):
    indexes = np.cumsum(facennodes)
    faces=np.zeros((indexes.shape[0]-1,np.amax(facennodes)),dtype=np.int32)
    for i in range(indexes.shape[0]-1):
        faces[i]=(facenodes[indexes[i]:indexes[i+1]])
    return faces


@njit(fastmath=True)
def makeCellsList(faces,c0,c1):
    cells = np.zeros((max(np.amax(c1),np.amax(c0))+1,6),dtype=np.int32)
    c1 = np.append(c1,np.zeros(faces.shape[0]-c1.size,dtype=np.int32))

    for face,(cell0,cell1) in enumerate(zip(c0,c1)):
        
        ind = numbaFirstZero(cells[cell0])
        cells[cell0,ind] = face

        if cell1 !=0:
            ind = numbaFirstZero(cells[cell1])
            cells[cell1,ind] = face
    return cells,c1

@njit
def numbaFirstZero(a):
    """ function that returns the index of the first zero in a flat array """
    return np.where(a==0)[0][0]

@njit
def getNodesFromFaces(cellfaces,facelist,nodelist):
    """ function that grabs all the nodes from a list of faces, typically the faces of a cell """
    faces = facelist[cellfaces].flatten()
    nodes = nodelist[faces]
    return nodes

@njit(fastmath=True)
def getCellCenters(cellNodes,dim=3):
    """ function that computes the center of all cells """
    ccenters = np.zeros((cellNodes.shape[0],dim),dtype=np.float64)
    for i in range(cellNodes.shape[0]):
        ccenters[i] = numbaMean(cellNodes[i])
    return ccenters

@njit(parallel=True,fastmath=True)
def getCellCentersPar(cellNodes,dim=3):
    """ function that computes the center of all cells """
    ccenters = np.zeros((cellNodes.shape[0],dim),dtype=np.float64)
    for i in prange(cellNodes.shape[0]):
        ccenters[i] = numbaMean(cellNodes[i])
    return ccenters

@njit(fastmath=True)
def numbaMean(array,axis=0):
    """ function that computes the mean of an array of points"""
    return array.sum(axis=axis)/array.shape[axis]

@njit(fastmath=True)
def getCellNodes(cells,faces,nodes):
    """ function that returns the nodes indexed by cell number (0-n_cells) """
    cellNodes = np.zeros((cells.shape[0],cells.shape[1]*faces.shape[1],nodes.shape[1]),dtype=np.float64)
    for cell in range(cells.shape[0]):
        cellNodes[cell] = getNodesFromFaces(cells[cell],faces,nodes)
    return cellNodes

@njit
def getFaceCenters(faces,nodes):
    """ function that returns the centers of all faces """
    faceCenters = np.zeros((faces.shape[0],nodes.shape[1]),dtype=np.float64)
    for face in range(faces.shape[0]):
        faceCenters[face] = numbaMean(nodes[faces[face]])
    return faceCenters


if __name__ == "__main__":
    zone = 1
    with h5py.File('case.cas.h5', 'r') as f:
            facenodes = np.array(f[f'/meshes/{zone}/faces/nodes/1/nodes'])-1
            facennodes = np.array(f[f'/meshes/{zone}/faces/nodes/1/nnodes'])
    

