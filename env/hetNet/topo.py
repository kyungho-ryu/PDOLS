# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 21:15:35 2021

@author: Gachon
"""
import copy, heapq
import time
from collections import deque
from env.hetNet.util import set_heap
import networkx as nx
import matplotlib.pyplot as plt
class Topology():
    def __init__(self, _numNodes, _numScell, edge = []):
        self.numNodes = _numNodes #from macro(1) to ue..
        self.numMacro = 1
        self.numScell = _numScell
        self.numUe = _numNodes - (_numScell +1)
        self.edge = edge
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(tuple(range(_numNodes)))
        self.graph.add_edges_from(edge) #eg. (1, 4)
        self.mcellAccessLink = []
        self.possible_mcellU = 0
        #nx.draw(self.graph)
        #plt.show()
        
    def updateTopoWithEdge(self, e): #bi-directional edge, x_ij
        self.graph.add_edges_from(self.edge)
    
    def updateTopoWithWeight(self, dWeights): #weight[e]
        self.mcellAccessLink = []
        for e in dWeights:
            #print("e", e, dWeights[e])
            if (e[0], e[1]) in self.graph.edges :
                self.graph[e[0]][e[1]]['weight'] = dWeights[e]
            else :
                set_heap(self.mcellAccessLink, dWeights[e], e)

        self.mcellAccessLink = heapq.nlargest(self.possible_mcellU,self.mcellAccessLink)
        self.mcellAccessLink = [l[1] for l in self.mcellAccessLink]

    def updatePossible_mcellU(self, n):
        self.possible_mcellU = n


    def getShortestPath(self, s, d):
        if (s,d) in self.mcellAccessLink :
            path = [s,d]
        else :
            path = nx.shortest_path(self.graph, s, d, weight='weight', method='dijkstra')

        return path

    def setPathViaSeNBs(self, sources, d, _edge):
        G = nx.Graph()
        G.add_nodes_from(tuple(range(self.numNodes)))

        G.add_edges_from(_edge)  # eg. (1, 4)
        for e in G.edges: G[e[0]][e[1]]['weight'] = self.graph[e[0]][e[1]]['weight']
        # print("t", self.graph.edges)
        # print("g", G.edges)
        #nx.draw(G, with_labels=True)
        # path = [p for p in nx.all_shortest_paths(G, s, d, weight='weight', method='dijkstra')]
        path = nx.multi_source_dijkstra(G, sources, d, weight='weight')

        return path


    def getPathViaSeNBs(self, s, d, _edge): #s: UE, d: macro 
        #print("src ", s, " dst ", d)
        #for i in _edge :
        #    if i[0] >25 :
        #        print("test", _edge)
        #        break

        # G = nx.DiGraph()
        # G.add_nodes_from(tuple(range(self.numNodes)))


        # G.add_edges_from(_edge) #eg. (1, 4)
        # for e in G.edges : G[e[0]][e[1]]['weight'] = self.graph[e[0]][e[1]]['weight']
        #print("t", self.graph.edges)
        #print("g", G.edges)

        #path = [p for p in nx.all_shortest_paths(G, s, d, weight='weight', method='dijkstra')]
        path = nx.shortest_path(self.graph, d, s, weight='weight', method='dijkstra')
        length = nx.shortest_path_length(self.graph, d, s, weight='weight', method='dijkstra')

        print("g", self.graph.edges(data=True))
        print("path", path)
        print("length", length)

        # macro length
        # m_length = self.graph[d][s]['weight']
        # print("m_length", m_length)

        # if m_length > length : return path
        # else : return [d, s]

        return path

    def getNextHop(self, s, d):
        path = nx.shortest_path(self.graph, s, d)
        if len(path) > 1:
            return path[1];
        else:
            return -1;

    def getNextHops(self, s, d, margin):
        path = self.getMultiPath(s, d, margin)
        n = set();
        for p in path:
            n.add(p[1])
        return list(n);
        

