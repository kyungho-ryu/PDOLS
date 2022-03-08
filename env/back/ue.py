# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 23:13:24 2021

@author: Gachon
"""
from env.hetNet.node import *

class UE(Node):
    def __init__(self, _TxPower = 23, _RxPower = -106, _prbSchedule = True):
    	self.uid = None;
    	self.gid = None;	#eNB
    
    	self.maxCellRadius = 0;
    	self.TxPower = _TxPower;	#dBm
    	self.RxPower = _RxPower;	#dBm
    	self.assEnb = None
    	self.flowRate = 0;
    	self.prbSchedule = _prbSchedule;
    	self.bottleNec = None;
    
    	self.adjEnbs = set();	
    	self.adjUes = set();
    	self.anLinks = [];
    	self.infUEs = set();
    
    def addInfNodes(self, n, inf): #UE* n, double inf
        n.RxPower = inf; 
        self.infUEs.add(n);
    def addAdjEnbs(self, n): #eNB* n
        self.adjEnbs.add(n);
    def addAdjUes(self, n): #UE* n
        self.adjUes.add(n);
    def addAnLinks(self, l): #Link* l
        self.anLinks.append(l);