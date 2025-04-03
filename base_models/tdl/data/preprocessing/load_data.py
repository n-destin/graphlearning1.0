
import gudhi
import gudhi.simplex_tree
import torch
from collections import defaultdict
from torch import Tensor
import graph_tool
import graph_tool.topology as topology
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import itertools
from ..cochain import Cochain
from ..complex import Complex, BatchComplex


def get_structure(max_structure_dimension, graph, type = "ring", induced = True):
    # returns the rings in the graph up to a certain size

    rings = set()
    sorted_rings = set()
    for dimension in max_structure_dimension:
        edge_index = list()
        for node in range(dimension): # construct a path or a cell
            next_node = node + 1 if node < max_structure_dimension else 0
            appending = type == "ring" or node < next_node != 0
            if appending:
                edge_index.append([node, next_node])
            subgraph = graph_tool.Graph()
            subgraph.add_edge_list(edge_index)
        isomorphisms = topology.subgraph_isomorphism(subgraph, graph, induced=induced, subsgraph = True, generator=True)
        
        for subgraph in isomorphisms:
            node_mapping = subgraph.mapping 
            if induced and sorted(subgraph) not in sorted_rings:
                sorted_rings.add(tuple(sorted(subgraph)))
                rings.add(subgraph)
            else:
                appending = True
                for index in range(len(subgraph)):
                    if tuple(subgraph[index : ] + subgraph[0:index]) in rings:
                        appending = False 

                if appending:
                    rings.append(subgraph)
    return rings

# check if the grap constructed is isomorphic to any structure in the graph

def get_ring_boundaries(ring, triangulate = False):
    ''''use this function if you want to triangulate the ring'''
    ring = list(ring)
    if len(ring) <= 1:
        return
    boundaries = list()

    for bound_dimension in range(len(ring)):
        boundaries += [tuple(boundary) for boundary in itertools.combinations(ring, bound_dimension)]
    if triangulate:
        return boundaries
    return boundaries[-1] # the path complexes

def graph_to_simplex_tree(edge_index : Tensor, n_nodes : int, rings : int, paths : int, triangulate = False):
    '''simplex tree with only nodes and edges'''
    assert edge_index.shape(0) == 2

    # add nodes
    tree = gudhi.SimplexTree()
    for node in range(n_nodes):
        tree.insert([node])
    # add edges 
    for index in range(edge_index.shape(1)):
        tree.insert([[edge_index[0][index], edge_index[1][index]]])
    
    graph = graph_tool.Graph(directed=False)
    graph.add_edge_list(edge_index.T.tolist()) 

    additional_structures = list()

    if rings != None:
        rings = get_structure(min(rings, n_nodes), graph, "ring")
        additional_structures += [rings]
    if paths != None:
       paths = get_structure(min(n_nodes, paths), graph, "path")
       additional_structures += paths
    for structure in additional_structures:
        tree.insert(structure)

        # boundaries are alreay appended into the structures, unless triangulate = True. TODO: uncommend the code below for triangulations
        # boundaries = get_ring_boundaries(structure) # not a triangulation
        # for boundary in boundaries:
        #     tree.insert(boundary) # if the s
    return tree

def get_simplex_id(id_map, simplex):
    dimension = len(simplex)
    returning = id_map[dimension][simplex]

    return returning if returning else None


def get_boundaries(simplex): # return boundaries of a simplex
    return [get_simplex_id(tuple(combination)) for combination in itertools.combinations(simplex, len(simplex) - 1)]

def simplex_ids(simplex_tree, n_nodes):

    ''''''
    dimension = simplex_tree.dimension()
    simplex_id_map = [defaultdict(list)] * dimension # it is possible to have dimensions with no cochains

    simplices = [[]] * dimension
    simplices[0] = [tuple(node) for node in range(n_nodes)] # add vertices

    simplex_id_map[0] = {tuple(node) : node for node in range(n_nodes) } # add vertices

    for simplex in simplex_tree.get_simplices():

        # populate simplex_id mapping
        sim_dimension = len(simplex)
        new_id = simplex_id_map[dimension - 1] 
        simplex_id_map[sim_dimension - 1][simplex] = len(new_id)

        # add the simplex to simplices 
        simplices[sim_dimension] += [simplex]

    return simplex_id_map, simplices

def boundary_relations(simplices : list, simplex_id_map : list):
    ''' returns the boundaries and co_boundaries relations'''

    boundaries = [{}] * len(simplices)
    co_boundaies = [{}] * len(simplices)

    for dimension in len(simplices):
        for simplex in simplices[dimension]: 
            simp_id = get_simplex_id(simplex)
            # note that for rings, their boundaries are paths
            simp_boundaries = get_boundaries(simplex) 

            if dimension > 0 and boundaries:
                boundaries[dimension][simp_id] += simp_boundaries
            
            for boundary in simp_boundaries:
                co_boundaies[dimension - 1][boundary] += [simp_id]

    return boundaries, co_boundaies

def initialize_2d(length):
    return [[]] * length

def simplex_relations(boundaries : list, co_boundaries : list):
    '''
    constructs upper_index, lower_index, shared_boundaries, and co_shared_boundaries for a certain dimension
    '''
    complex_dimension = len(boundaries)
    upper_index, lower_index = initialize_2d(complex_dimension), initialize_2d(complex_dimension)
    shared_boundaries, shared_co_boundaires = initialize_2d(complex_dimension), initialize_2d(complex_dimension)
    

    for dimension in range(len(boundaries)): # for each dimension
        if dimension > 0:
            for simplex, boundaries_ in boundaries[dimension]: # get the simplex and its boundaries, and connect each combination of boundaries in upper_index
                for b_one, b_two in itertools.combinations(boundaries_):
                    upper_index[dimension - 1] += [[b_one, b_two]] # each two boundaries are upper connected
                    shared_boundaries[dimension - 1] += [simplex] # the shared boundary

        if dimension < complex_dimension:
            for simplex, co_boundaries in co_boundaries[dimension]:
                for co_b_one, co_b_two in itertools.combinations(co_boundaries):
                    lower_index[dimension + 1] += [co_b_one, co_b_two]
                    shared_co_boundaires[dimension + 1] += [simplex]
        
    
    return upper_index, lower_index, shared_boundaries, shared_co_boundaires


def consturct_complex(data):

    simplex_tree = graph_to_simplex_tree(data.edge_index)
    simplices, simplex_id_map = simplex_ids(simplex_tree, len(data.x)) # simplices :: List -> List -> simplices, simplices_id_map -> Map of simplex to id
    boundaries, co_boundaries = boundary_relations(simplices, simplex_id_map)

    all_upper_index, all_lower_index, all_shared_boundaries, all_shared_co_boundaries = simplex_relations(boundaries, co_boundaries)
    
    dimension = len(boundaries) # dimension of the complex

    cochains = list() 
    for dim in range(dimension):

        boundary_index = [[boundaries[node], [node] * len(dimension[node])] for node in boundaries[dim]]

        upper_index = all_upper_index[dimension]
        lower_index = all_lower_index[dimension]
        shared_boundaries = all_shared_boundaries[dimension] # index of simplex shared by lower_index connections
        shared_co_boundaries = all_shared_co_boundaries[dimension] # index of simplex shared by upper_index connections

        cochain = Cochain(dim, None, upper_index, lower_index, boundary_index, shared_boundaries, shared_co_boundaries)
        cochains.apend(cochain)

    complex = Complex(cochains, dimension)

    return complex


def construct_complexes(data_list):

    complexes = list()
    for data in data_list:
        complex = consturct_complex(data)
        complexes.appen (complex)

    return BatchComplex(complexes)