# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:15:26 2021

@author: Gachon
"""

MAX_ROUTE_VIA_GW = 10
class Node:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.ch = 0
        self.id = 0
        self.bSource = None
        self.adjNodes = set()
        self.infNodes = set()
        self.adjLinks = []
        self.routeMap = {}
        self.routeMapVia = {}
        self.bVisit = False
        self.txPower = 0
        self.numViaGw = 0
        self.optimalGw = 0
    def addAdjNodes(n):
        self.adjNodes.add(n)
    def addInfNodes(n):
        infNodes.add(n)