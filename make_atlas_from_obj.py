#!/usr/bin/env python3

from os.path import join

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#import pywavefront
#from pywavefront import visualization as vis
import pyglet
#import pymesh
import trimesh


# https://stackoverflow.com/questions/47974874
from sklearn.cluster import DBSCAN

def cluster(data, epsilon, N): #DBSCAN, euclidean distance
    db     = DBSCAN(eps=epsilon, min_samples=N).fit(data)
    labels = db.labels_ #labels of the found clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) #number of clusters
    clusters   = [data[labels == i] for i in range(n_clusters)] #list of clusters
    return clusters, n_clusters


import networkx as nx
import scipy.spatial as sp

class IGraph:
    def __init__(self, nodelst=[], radius = 1):
        self.igraph = nx.Graph()
        self.radii  = radius
        #nodelst is array of coordinate tuples, graph contains indices as nodes
        self.nodelst = nodelst
        self.__make_edges__()

    def __make_edges__(self):
        self.igraph.add_edges_from( sp.cKDTree(self.nodelst).query_pairs(r=self.radii) )

    def get_conn_comp(self):
        ind = [list(x) for x in nx.connected_components(self.igraph) if len(x)>1]
        return [self.nodelst[indlist] for indlist in ind]


def graph_cluster(data, epsilon):
    graph = IGraph(nodelst = data, radius = epsilon)
    clusters = graph.get_conn_comp()
    return clusters, len(clusters)


def main():
    obj_path = join(
        'cs-transfer', 'Merged_2-101221a-labels_only_sure_ones_Sensillarcolors.obj'
    )

    #window = pyglet.window.Window()

    #obj = pywavefront.Wavefront(obj_path, strict=True, cache=True, collect_faces=True)

    #vis.draw(obj)
    #pyglet.app.run()

    mesh = trimesh.load(obj_path)


    '''
    mesh = pymesh.load_mesh(obj_path)

    pts,_ = pymesh.mesh_to_graph(mesh)
    x,y,z=zip(*pts)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    fig = plt.figure(figsize=(800/72,800/72))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='Greens')
    #plt.show()
    '''

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)


    # TODO implement a binary search for each digit (or something less stupid?) to find
    # argument producing minimum nonzero value of # of clusters
    # lowest i've observed so far is 109 clusters at 0.019199999999999
    # 0.0191999999999999 -> 111
    # 0.019199999999 - > 0
    # NOTE: couldn't actually figure out how to use the output of this, or whether it
    # was useful at all
    clusters = trimesh.grouping.clusters(vertices, 0.0192)

    '''
    def get_mesh(vertex_indices):
        # TODO how to know which faces to include? (diff length than list of vertices)
        Trimesh()
    '''

    #def mesh_from_connected_faces(start_vertex_index):
    #    vertex_faces = 

    n_clusters = len(clusters)

    '''
    # DBSCAN based method (min_samples=10)

    # 0
    #eps = 0.3
    # 100
    #eps = 0.5
    # 227
    #eps = 0.6

    # 587
    # 291062 vertices clustered
    #eps = 1.0

    # all vertices clustered
    eps = 2.0

    min_samples = 10

    clusters, n_clusters = cluster(vertices, eps, min_samples)

    #clusters2, n_clusters2 = graph_cluster(vertices, eps)
    #assert n_clusters == n_clusters2, f'{n_clusters} != {n_clusters2}'

    print('eps:', eps)
    print('min_samples:', min_samples)
    '''

    print('n_clusters:', n_clusters)

    print('# vertices:', len(vertices))
    print('# clustered vertices:', sum([len(xs) for xs in clusters]))

    def n_total_of(meshes, of):
        return sum([len(getattr(mesh, of)) for mesh in meshes])

    def n_total_vertices(meshes):
        return n_total_of(meshes, 'vertices')

    def n_total_faces(meshes):
        return n_total_of(meshes, 'faces')

    def print_n_vertices_and_faces(meshes_or_mesh):

        if isinstance(meshes_or_mesh, trimesh.Trimesh):
            meshes = [meshes_or_mesh]
        else:
            meshes = meshes_or_mesh

        n_vertices = n_total_vertices(meshes)
        n_faces = n_total_faces(meshes)

        print(f'{n_vertices} vertices')
        print(f'{n_faces} faces')

        return n_vertices, n_faces

    print()
    print('Original mesh:')
    n_orig_vertices, n_orig_faces = print_n_vertices_and_faces(mesh)
    print()

    split_meshes = mesh.split(only_watertight=False)
    # NOTE: this has slightly more vertices / faces in total, presumably because
    # duplication across meshes (vertex adjacency w/o face adjacency?)
    print('Across all meshes from split (only_watertight=False):')
    n_total_split_vertices, n_total_split_faces = print_n_vertices_and_faces(
        split_meshes
    )
    print()

    # This has basically nothing (5, 6)
    #print('Across all meshes from split (only_watertight=True):')
    #n_total_split_vertices, n_total_split_faces = print_n_vertices_and_faces(
    #    mesh.split(only_watertight=True)
    #)
    #print()

    orig_mesh = mesh

    def get_set_of(mesh, of):
        return {tuple(x) for x in getattr(mesh, of)}

    def get_vertex_set(mesh):
        return get_set_of(mesh, 'vertices')

    def get_face_set(mesh):
        face_index_set = list(get_set_of(mesh, 'faces'))
        face_set = {tuple(tuple(mesh.vertices[v]) for v in vs) for vs in face_index_set}
        return face_set

    '''
    # Checking we don't lose any specific vertices or faces
    orig_only_vertex_set = get_vertex_set(orig_mesh)
    # NOTE: there are a few duplicate faces it seems
    # (721540 in mesh.faces and 721533 here)
    orig_only_face_set = get_face_set(orig_mesh)

    for mesh in split_meshes:
        orig_only_vertex_set -= get_vertex_set(mesh)
        orig_only_face_set -= get_face_set(mesh)

    assert len(orig_only_vertex_set) == 0
    assert len(orig_only_face_set) == 0
    '''

    def show(*mesh, block=True, **kwargs):

        if len(mesh) == 0:
            # Same thing trimesh does if start_loop=True
            pyglet.app.run()
            return
        else:
            assert len(mesh) == 1
            mesh = mesh[0]

        mesh.show(smooth=False, start_loop=block, **kwargs)

    for mesh in split_meshes:
        # TODO why are there some meshes w/ length 1 mesh.vertices?
        if len(mesh.vertices) == 1:
            continue

        # Since some of the "repair" operations change the mesh.
        orig = mesh.copy()

        mesh.fix_normals()
        mesh.fill_holes()

        for m, name in ((orig, 'orig'), (mesh, 'repaired')):
            print(name)
            print(f'watertight? {m.is_watertight}')
            print_n_vertices_and_faces(m)
            print()

            show(m, block=False, caption=name)

        show()

        import ipdb; ipdb.set_trace()
        print()

    show(orig_mesh)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

