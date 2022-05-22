class Case:
    def __init__(self,filename) -> None:
        from ANSYSfluent import readCase as rc
        from ANSYSfluent import getMeshes as gm
        self.meshes = gm(filename)
        if len(self.meshes)==1:
            self.nodes,self.faces,self.cells,self.faceCenters,self.cellCenters,self.c0,self.c1,self.zoneId,self.zonetype,self.minId,self.maxId,self.surfaces = rc(filename,int(self.meshes),returnMore=True)
        else:
            print(f'case has multiple meshes:\n {self.meshes}')
            for mid in self.meshes:
                exec("self.mesh%s = Mesh(filename,int(%s))"%(mid,mid))
    
    def __str__(self) -> str:
        if len(self.meshes)==1:
            output =f'Amount of cells: {len(self.cells)} \nAmount of faces: {len(self.faces)} \nAmount of nodes: {len(self.nodes)}\n'
            for id,name in self.surfaces.items():
                output += f'Surface {id} has name {name}\n'
            return output
        
        else:
            return 'case has multiple meshes'
    
    def plot_meshplot(self,colorDict=None):
        import meshplot as mp
        from ANSYSutils import quadToTria,triaColors
        if len(self.meshes)==1:
            
            boundId = (self.c1==0)  #this makes sure that we only plot the boundaries

            triangles = quadToTria(self.faces[boundId])

            colors = triaColors(self.faces,self.minId,self.maxId,self.zonetype,boundId,colorDict)

            return mp.plot(self.nodes,triangles,colors)

    def plot_ipv(self,colorDict=None):
        import ipyvolume as ipv
        from ANSYSutils import quadToTria,triaColors,faceToNodeColors

        boundId = (self.c1==0)  #this makes sure that we only plot the boundaries

        triangles = quadToTria(self.faces[boundId])
        colors = triaColors(self.faces,self.minId,self.maxId,self.zonetype,boundId,colorDict)

        nodeColors = faceToNodeColors(triangles,colors,self.nodes.shape)
    
        ipv.plot_trisurf(self.nodes[:,0], self.nodes[:,1], self.nodes[:,2],triangles,color=nodeColors )
        ipv.show()

class Mesh:
    def __init__(self,filename,mesh) -> None:
        from ANSYSfluent import readCase as rc
        self.nodes,self.faces,self.cells,self.faceCenters,self.cellCenters,self.c0,self.c1,self.zoneId,self.zonetype,self.minId,self.maxId,self.surfaces = rc(filename,mesh,returnMore=True)