# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:49:07 2021

@author: Gachon
"""
LOG_TYPE_UE = 1;
LOG_TYPE_ENB = 2;

logType = LOG_TYPE_UE | LOG_TYPE_ENB;

class simLog:
    def __init__(self, s):
        self.sim = s
        self.ueLoc = {};
        self.macroLoc = None
        self.smallLoc = {}
        self.bh = {}
        self.an = {}
        self.logf = open("log", mode='wt')

    def MSG_LOG(self, t, msg):
        if logType & t:
            self.logf.write(msg+"\n")
            
    def saveResult(self):
        f = open("result", mode='w')
        f.write("MeNB location: "+ str(self.macroLoc) + "\n")
        f.write("SeNB location: "+ str(self.smallLoc) + "\n")
        f.write("UE location: "+ str(self.ueLoc) + "\n")
        f.write("backhaul: "+ str(self.bh) + "\n")
        f.write("access link: "+ str({k: self.sim.scellAcessLink[k].maxCapa for k in self.sim.scellAcessLink}) + "\n")
        f.write("UE dataRate: "+ str({u.id:u.flowRate for u in self.sim.ues}) + "\n")
        f.close();
        
    def closeLog(self):
        self.logf.close()