from numba import njit
import numpy as np

@njit
def quadToTria(faces):
    """
    converts quadrilateral faces to triangular faces
    faces: numpy array of quad faces (n,4) containing the indices of the nodes/vertices
    THIS FUNCTION PRESERVES THE RIGHT HAND RULE

    """
    if faces.shape[1] != 4:
        raise ValueError('faces must be a numpy array of quad faces (n,4)')

    trifaces = np.zeros((faces.shape[0]*2,3),dtype=np.int32)
    for i,[n1,n2,n3,n4] in enumerate(faces):
        trifaces[2*i]  = np.array([n1,n2,n3])
        trifaces[2*i+1] = np.array([n1,n3,n4])
    return trifaces

def triaColors(faces, minId,maxId,zoneType,boundId,colordict=None):
    """
    assigns colors based on the type of boundary
    faces : numpy array of triangular faces (n,3)
    minId : minimum id of the zone
    maxId : maximum id of the zone
    zoneType : ANSYS zone type (inlet,outlet,wall,symmetry,...)
    boundId : arrays with shape = faces.shape and contains true where the boundaries are, this can be obained with np.where(c1==0).
    colordict: dictionary of colors {int: np.array([r,g,b],dtype=np.floatxx)} with r,g,b in [0,1]
    maxId,minId,zoneType,boundId are all outputs of readCase if you specify the parameter returnMore=True
    output: numpy array of colors (n,3) based on the type of boundary specified in Zonetype
    """
    default       = np.array([0.5,0.5,0.5],dtype=np.float64)        #gray
    inletcolor    = np.array([1,0,0],dtype=np.float64)              #red
    wallcolor     = np.array([0.25,0.25,0.25],dtype=np.float64)     #black
    symmetrycolor = np.array([0,0,1],dtype=np.float64)              #green
    outletcolor   = np.array([0,1,0],dtype=np.float64)              #blue
    
    if  isinstance(colordict,type(None)):
        colordict = {2:default,3:wallcolor,4:inletcolor,5:outletcolor,7:symmetrycolor,8:default,9:default,10:inletcolor,11:default,12:default,13:default,14:inletcolor}
    
    c = np.zeros((faces.shape[0],3),dtype=np.float64)
    for i,z in enumerate(zoneType):
        c[minId[i]:maxId[i]] = colordict[z]
    
    c = c[boundId]
    
    cfinal = np.zeros((c.shape[0]*2,*c.shape[1:]),dtype=np.float64)
    for i,col in enumerate(c):
        cfinal[2*i]   = col
        cfinal[2*i+1] = col
    
    return cfinal