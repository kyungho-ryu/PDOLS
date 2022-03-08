# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:12:51 2021

@author: Gachon
"""
from env.hetNet.node import *
MACRO_CELL_RADIUS =500
SMALL_CELL_RADIUS =50
BACKHAUL_60_DISTANCE =200

ENB_TYPE_MACRO = 0
ENB_TYPE_SMALL = 1

class eNB(Node):
    def __init__(self, aTxPower= 43, bTxPower= 43, t= ENB_TYPE_MACRO, r= MACRO_CELL_RADIUS):
        self.AnEirp = aTxPower;
        self.BhEirp = bTxPower;
        self.AnTxPower = self.AnEirp;
        self.BhTxPower = self.BhEirp;
        self.enbType = t;
        self.maxCellRadius = r;
        self.maxBhDist = BACKHAUL_60_DISTANCE; 
        self.gid =0;
        self.numPRB = 0;
        self.allocPrb=0;
        self.adjEnbs = set()
        self.adjUEs = set()
        self.assUEs = set()
        self.infEnbs = set()
        self.bhLinks = []
        self.anLinks = []
        #Link* assLinks;	//associated links
    def addAdjNodes(self, n):
        self.adjEnbs.add(n)
    def addInfNodes(self, n):
        self.infEnbs.add(n)
    def addAdjLinks(self, l):
        self.bhLinks.append(l)
    def addAdjUEs(self, u):
        self.adjUEs.add(u)
    def addAssUEs(self, u):
        self.assUEs.add(u)
    def addBhLinks(self, l):
        self.bhLinks.append(l)
    def addAnLinks(self, l):
        self.anLinks.append(l)  
    def getTxPower(n): #Node* n
        if n in self.adjUEs:
            return self.AnTxPower;
        elif n in self.adjEnbs:
            return self.BhTxPower;
#	double getAnTxPower(){	return AnTxPower;	}
#	double getBhTxPower(){	return BhTxPower;	}
#	double getAnEirp() { return AnEirp; }
#	double getBhEirp() { return BhEirp; }
#	double getAnDist(){ return maxCellRadius; }	//acc link distance
#	double getBhDist(){ return maxBhDist; }
#	double getRxPower() { return (-80); }	//have to change for uplink per each UE
#	virtual double getTxPower(Node* n);
#	int getNumAssUEs() { return assUEs.size(); }
#	void addAssLinks(Link* l) { assLinks =l; }
#	Link* getAssLink() { return assLinks; }
#	
#	
#	std::set<eNB*> getAdjEnb() { return adjEnbs; }
#	void setEnbType(ENB_TYPE t){ enbType = t; }
#	#ENB_TYPE getEnbType(){ return enbType; }
#	bool findUE(Node* n);
#	bool findEnb(Node* n);
#	void setAllocPrb(int p) { allocPrb = p; }
#	int getAllocPrb() { return allocPrb; }
#	std::list<Link*> getAnLinks(){ return anLinks; }
#	std::list<Link*> getBhLinks(){ return bhLinks; }
#	std::set<UE*> getAssUes() { return assUEs; }
#	void TxPowerAllocation();
