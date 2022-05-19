import numpy as np
import h5py
from numba import njit,prange
from Case import Case

def case(filename):
    return Case(filename)

def readCase(filename,zone=1,returnMore=False):
    """ function that reads a case file and returns the relevant mesh data of the zone"""

    with h5py.File(filename, 'r') as f:
        nodes = np.array(f[f'/meshes/{zone}/nodes/coords/{zone}'])
        facenodes = np.array(f[f'/meshes/{zone}/faces/nodes/{zone}/nodes'])-1
        facennodes = np.array(f[f'/meshes/{zone}/faces/nodes/{zone}/nnodes'])
        c0 = np.array(f[f'/meshes/{zone}/faces/c0/{zone}'])-1
        c1 = np.array(f[f'/meshes/{zone}/faces/c1/{zone}'])-1

        zoneId = np.array(f[f'/meshes/{zone}/faces/zoneTopology/id'])
        minId = np.array(f[f'/meshes/{zone}/faces/zoneTopology/minId'])
        maxId = np.array(f[f'/meshes/{zone}/faces/zoneTopology/maxId'])
        zonetype = np.array(f[f'/meshes/{zone}/faces/zoneTopology/zoneType'])
    
    # nodes = np.insert(nodes,0,np.array([0,0,0],dtype=np.float64),axis=0)

    ind = 0
    # faces = [np.array([0,0,0,0])]
    faces = []
    for n in facennodes:
        faces.append(facenodes[ind:ind+n])
        ind += n

    faces = np.array(faces)

    cells = np.zeros((max(np.amax(c1),np.amax(c0))+1,6),dtype=np.int32)

    for face,cell in enumerate(c0):

        try:
            ind = numbaFirstZero(cells[cell])
        except IndexError:
            print(f'the faces of cell{cell}:',cells[cell])
            print('fucntion output ',np.where(cells[cell]==0))
        cells[cell,ind] = face

    for face,cell in enumerate(c1):
        if cell !=0:
            ind = numbaFirstZero(cells[cell])
            cells[cell,ind] = face

    cellCenters = getCellCenters(getCellNodes(cells,faces,nodes))
    faceCenters = getFaceCenters(faces,nodes)

    if returnMore:
        return nodes,faces,cells,faceCenters,cellCenters,c0,c1,zoneId,zonetype,minId,maxId

    return nodes,faces,cells,faceCenters,cellCenters

@njit
def numbaFirstZero(a):
    """ function that returns the index of the first zero in an array """
    return np.where(a==0)[0][0]

@njit
def getNodesFromFaces(cellfaces,facelist,nodelist):
    """ function that grabs all the nodes from a list of faces, typically the faces of a cell """
    faces = facelist[cellfaces].flatten()
    nodes = nodelist[faces]
    return nodes

@njit(fastmath=True)
def getCellCenters(cellNodes):
    """ function that computes the center of all cells """
    ccenters = np.zeros((cellNodes.shape[0],3),dtype=np.float64)
    for i in range(cellNodes.shape[0]):
        ccenters[i] = numbaMean(cellNodes[i])
    return ccenters

@njit(parallel=True,fastmath=True)
def getCellCentersPar(cellNodes):
    """ function that computes the center of all cells """
    ccenters = np.zeros((cellNodes.shape[0],3),dtype=np.float64)
    for i in prange(cellNodes.shape[0]):
        ccenters[i] = numbaMean(cellNodes[i])
    return ccenters

@njit(fastmath=True)
def numbaMean(array,axis=0):
    """ function that computes the mean of an array """
    return array.sum(axis=axis)/array.shape[axis]

@njit(fastmath=True)
def getCellNodes(cells,faces,nodes):
    """ function that returns the nodes indexed by cell number (0-n_cells) """
    cellNodes = np.zeros((cells.shape[0],cells.shape[1]*faces.shape[1],3),dtype=np.float64)
    for cell in range(cells.shape[0]):
        cellNodes[cell] = getNodesFromFaces(cells[cell],faces,nodes)
    return cellNodes

@njit
def getFaceCenters(faces,nodes):
    """ function that returns the centers of all faces """
    faceCenters = np.zeros((faces.shape[0],3),dtype=np.float64)
    for face in range(faces.shape[0]):
        faceCenters[face] = numbaMean(nodes[faces[face]])
    return faceCenters


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    nodes,faces,cells,faceCenters,cellCenters=readCase('case.cas.h5')
    print('elapsed time: ',time.perf_counter()-t,' seconds')
