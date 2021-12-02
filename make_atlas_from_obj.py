#!/usr/bin/env python3

from os.path import join

import numpy as np
import pyglet
import trimesh
from trimesh import PointCloud


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
        'from_veit', 'Merged_2-101221a-labels_only_sure_ones_Sensillarcolors.obj'
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


    # TODO implement a binary search for each digit (or something less stupid?) to find
    # argument producing minimum nonzero value of # of clusters
    # lowest i've observed so far is 109 clusters at 0.019199999999999
    # 0.0191999999999999 -> 111
    # 0.019199999999 - > 0
    # NOTE: couldn't actually figure out how to use the output of this, or whether it
    # was useful at all
    clusters = trimesh.grouping.clusters(vertices, 0.0192)

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
    del vertices


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
        else:
            assert len(mesh) == 1
            mesh = mesh[0]

            if isinstance(mesh, trimesh.Trimesh):
                flags = {'wireframe': True}
            else:
                flags = None

            #mesh.show(smooth=False, start_loop=block, flags=flags, **kwargs)

    transparency_rgb = tuple([x for x in orig_mesh.visual.face_colors[0][:3]])
    # 0.4 was OK for mesh but seems way too high for points (unless there's another
    # problem)
    alpha = 0.4
    alpha_u8 = int(round(alpha * 255))
    transparency_rgba = transparency_rgb + (alpha_u8,)

    orig_mesh.visual.face_colors = transparency_rgba

    focus_u8_alpha = 190
    focus_color_rgba = (255, 0, 0, focus_u8_alpha)
    shared_vertex_color_rgba = (0, 255, 0, focus_u8_alpha)

    vertex2orig_idx = {tuple(x): i for i, x in enumerate(orig_mesh.vertices)}
    def convert_face_indices_to_orig(mesh):
        return np.array([[vertex2orig_idx[tuple(x)] for x in mesh.vertices[face]]
            for face in mesh.faces
        ])

    face2face_idx = {tuple(x): i for i, x in enumerate(orig_mesh.faces)}
    def get_orig_faces_mask(orig_face_indices):
        mask = np.ones(len(orig_mesh.faces), dtype=np.bool_)
        masked_indices = [face2face_idx[tuple(face)] for face in orig_face_indices]
        mask[masked_indices] = False
        return mask


    print(f'{len(split_meshes)} meshes after splitting with face adjacencies')

    #from pprint import pprint
    #pprint(sorted([len(x.vertices) for x in split_meshes]))
    #pprint(sorted([len(x.faces) for x in split_meshes]))

    # NOTE: there is also a fair bit of stuff with 2, 3, and other low numbers of faces
    # (with no obvious sharpj dropoff in counts at a particular value)
    min_faces = 2

    split_meshes = [x for x in split_meshes if len(x.faces) >= min_faces]
    print(f'{len(split_meshes)} split meshes after removing those with <{min_faces}'
        ' faces\n'
    )

    # TODO TODO probably build a data structure of points -> faces/bodies that contain
    # them. want to be able to search for additional faces to merge into ROIs w/ a big
    # missing component.

    for i, mesh in enumerate(split_meshes):

        #print(f'watertight? {mesh.is_watertight}')
        n_vertices, n_faces = print_n_vertices_and_faces(mesh)

        curr_vertex_set = {tuple(x) for x in mesh.vertices}

        minus_curr = orig_mesh.copy()
        orig_face_indices = convert_face_indices_to_orig(mesh)
        mask = get_orig_faces_mask(orig_face_indices)
        # TODO TODO is it just some artifact of how i'm generating this that the
        # interior face i expected (thought to be shared by some neighboring glomeruli)
        # just isn't there at all??? (seems so)
        # TODO TODO maybe comparison to plots of the vertices alone could help clarify?
        minus_curr.update_faces(mask)

        minus_curr.remove_unreferenced_vertices()

        minus_curr_vertex_set = {tuple(x) for x in minus_curr.vertices}

        # TODO maybe also show this just w/in bounds defining 3d bbox
        #show(minus_curr, caption='everything except current mesh', block=False)

        n_less_vertices = len(orig_mesh.vertices) - len(minus_curr.vertices)
        n_less_faces = len(orig_mesh.faces) - len(minus_curr.faces)
        print()
        print('after filtering current mesh faces from everything:')
        # NOTE: this is (at least in first example) less than the number of vertices
        # in mesh. implies some of these vertices participate in faces that would get
        # split into other meshes...
        print(f'{n_less_vertices} less vertices')
        print(f'{n_less_faces} less faces')

        #show(PointCloud(mesh.vertices, colors=focus_color_rgba), block=False,
        #    caption='current mesh vertices'
        #)

        beyond_bounds = 0.0

        # TODO is "axis aligned" in `trimesh.bounds.contains` docs same as 'oriented'
        # here? and does it actually matter? the non-oriented version seems to have
        # bounds more like what i want / expect
        #bounds = np.array(mesh.bounding_box_oriented.bounds)
        bounds = np.array(mesh.bounding_box.bounds)

        bounds[0] -= beyond_bounds / 2
        bounds[1] += beyond_bounds / 2

        def in_bounds(vs):
            mask = trimesh.bounds.contains(bounds, vs)
            return vs[mask]

        vertices_in_curr_mask = np.array([
            tuple(x) in curr_vertex_set for x in orig_mesh.vertices
        ])

        shared_vertex_set = minus_curr_vertex_set & curr_vertex_set
        shared_vertex_mask = np.array([
            tuple(x) in shared_vertex_set for x in orig_mesh.vertices
        ])

        print(f'{len(shared_vertex_set)} shared vertices')

        shared_vertex_indices = set(np.where(shared_vertex_mask)[0])
        for j, other_mesh in enumerate(split_meshes):
            if i == j:
                continue

            # TODO maybe try starting w/ trimesh.graph.shared_edges rather than
            # computing shared points and trying to find faces from there

            other_orig_face_indices = convert_face_indices_to_orig(other_mesh)
            other_unq_vertex_indices = set(np.unique(other_orig_face_indices))

            overlap = other_unq_vertex_indices & shared_vertex_indices
            if len(overlap) == 0:
                continue

            print(f'{i}-{j}: {len(overlap)} of shared points')

            #import ipdb; ipdb.set_trace()

        import ipdb; ipdb.set_trace()

        full_vertex_colors = np.array([
            transparency_rgba for _ in range(len(orig_mesh.vertices))
        ])
        full_vertex_colors[vertices_in_curr_mask] = focus_color_rgba

        bound_mask = trimesh.bounds.contains(bounds, orig_mesh.vertices)
        bound_colors = full_vertex_colors[bound_mask]

        show(PointCloud(orig_mesh.vertices[bound_mask], colors=bound_colors),
            block=False, caption='all vertices'
        )

        full_vertex_colors[shared_vertex_mask] = shared_vertex_color_rgba
        bound_colors = full_vertex_colors[bound_mask]

        # NOTE: syntax for adding meshes before viewing didn't seem to work w/ point
        # clouds
        show(PointCloud(orig_mesh.vertices[bound_mask], colors=bound_colors),
            block=False, caption='all vertices (shared highlighted)'
        )

        vertices_minus_curr = orig_mesh.vertices[~ vertices_in_curr_mask]

        show(PointCloud(in_bounds(vertices_minus_curr), colors=transparency_rgba),
            block=False, caption='all vertices except from current mesh'
        )

        show(mesh, block=False)

        mesh.visual.face_colors = focus_color_rgba

        show(orig_mesh + mesh, caption='overlay', block=False)

        show()

        import ipdb; ipdb.set_trace()
        print()

    show(orig_mesh)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

