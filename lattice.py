import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from meshpy.tet import MeshInfo, build
from meshpy.geometry import GeometryBuilder

def writeHeader(outFile, commentLine, keywordLine):
    """Writes header on file object

    outFile, file object: already opened file object to write
    commentLine, str: string which goes in as comment
    keywordLine, str: keyword line with parameters (if required)
    Examples
    >>> header(testfile, "Nodal Coordinates", "NODE")
    >>> header(testfile, "Elements", "ELEMENT, TYPE=S4")
    """
    outFile.write("**\n")
    outFile.write("%s\n" %("*"*(len(commentLine) + 3)))
    outFile.write("** %s\n" %(commentLine))
    outFile.write("%s\n" %("*"*(len(commentLine) + 3)))
    outFile.write("*%s\n" %(keywordLine))

def writeNodeLine(outFile, nodeNumber, coords):
    """Writes the nodal coordinates line in Abq format, doesnt return anything

    outFile, file object: already opened file object to write
    nodeNumber, int: node number
    coords, list of float: list of coordinates of current node(<x1>,<x2>,<x3>)
    """
    outFile.write("%d, %g, %g, %g\n" %(nodeNumber, coords[0], coords[1], coords[2]))

# Similarly for Element definition
def writeElemLine(outFile, elemNumber, nodalCnvty):
    """Writes the Element line in Abq format, doesnt return anything; This is
    valid only for 4 noded shell elements.

    outFile, file object: already opened file object to write
    elemNumber, int: Element number
    nodalCnvty, list of int: Nodal connectivity of current element (<node1>,
                            <node2>, <node3>, <node4>)
    """
    outFile.write("%d, %d, %d, %d, %d\n" %(elemNumber, nodalCnvty[0], nodalCnvty[1], nodalCnvty[2], nodalCnvty[3]))

def surf_to_inp(surf_mesh, holes, filename):
    '''
    converts triangular surface mesh to tetrahedral volumetric mesh and saves as abaqus INP file format
    '''
    # build geometry using meshpy
    builder = GeometryBuilder()
    builder.add_geometry(points=surf_mesh.vertices.tolist(), facets=surf_mesh.faces.tolist())
    builder.wrap_in_box(1)
    
    # create tetrahedral mesh
    mi = MeshInfo()
    builder.set(mi)
    mi.set_holes(holes) # set holes
    mesh = build(mi)
    #print("%d elements" % len(mesh.elements))
    #mesh.write_vtk("out.vtk")
    
    # prepare for INP file format
    nodes = np.asarray(mesh.points)
    elems = np.asarray(mesh.elements) + 1 # node numbers start changed from 0 to 1

    with open(filename, "w") as inp:
        writeHeader(inp, "Nodal Coordinates", "NODE")    # Write node header
        # write vertex nodes
        for i  in range(nodes.shape[0]):
            writeNodeLine(inp, i+1, nodes[i])

        inp.write("\n")
        writeHeader(inp, "Elements", "ELEMENT, TYPE=C3D4")    # Write Element header
        # write tetrahedral elements
        for i  in range(elems.shape[0]):
            writeElemLine(inp, i+1, elems[i])
    print('Surf converted to INP!')

def field_to_surf(field, iso = 0., gridDim = [1., 1., 1.]):
    '''
    coverts from 3D field values to triangular surface mesh
    '''
    # add a padding of zeros to get boudary surfaces
    field = np.pad(field, 1, 'constant', constant_values = 0.)
    # marching cube for extracting iso-surface
    vertices, faces, normals, values = measure.marching_cubes_lewiner(field, iso, tuple(gridDim))
    # creating surf_mesh structure
    surf_mesh = trimesh.Trimesh(vertices, faces)
    print('Field converted to Surf!')
    return surf_mesh

def field_to_inp(field, filename, holes = [], iso = 0., gridDim = [1., 1., 1.]):
    '''
    converts field to abaqus INP file format
    field -> surface mesh -> volumetric mesh -> INP file
    '''
    surf_to_inp(field_to_surf(field, iso, gridDim), holes, filename)
    print('Field converted to INP!')
    
def field_to_stl(field, filename, iso = 0., gridDim = [1., 1., 1.]):
    '''
    converts field to STL surface mesh file format
    field -> surface mesh -> STL file
    '''
    surf_mesh = field_to_surf(field, iso, gridDim)
    surf_mesh.export(filename, 'stl_ascii')
    print('Field converted to STL!')
    
def stl_to_inp(stl_filename, inp_filename, holes = []):
    '''
    converts a surface mesh from STL file to volumetric abaqus INP file
    '''
    surf_to_inp(trimesh.load(stl_filename), holes, inp_filename)
    print('STL converted to INP!')
    