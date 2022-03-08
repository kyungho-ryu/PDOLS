# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 15:31:20 2021

@author: Gachon
"""
import random
import math
from env.hetNet.eNB import *
from env.hetNet.util import *
from env.hetNet.link import *
from env.hetNet.ue import *
from env.hetNet.channel import *
from env.hetNet.log import *
from env.hetNet.topo import *

random.seed(random.random())

BW_200MH =200000000
BW_400MH =400000000
BW_600MH =600000000
BW_800MH =600000000
BW_250MH =250000000
BW_500MH =500000000
BW_1000MH =1000000000
BW_2100MH =2100000000
BW_10MH =10000000
BW_20MH =20000000

BW10MH_PRB =50
BW20MH_PRB =100
BW1000MH_PRB =5000

MAX_NUM_ANT =4
AN_TX_ANT_GAIN =15
AN_RX_ANT_GAIN =7

MM60_TX_ANT_GAIN =36
MM60_RX_ANT_GAIN =36

MM80_TX_ANT_GAIN =43
MM80_RX_ANT_GAIN =43


RX_SENS =-107.5

MIN_UE_MACRO_DIST =35
MIN_UE_SMALL_DIST =7

MIN_ENB_MACRO_DIST =100
MIN_ENB_SMALL_DIST =20

MACRO_AN_TX_P =43 #46 dbm change according to conf. paper
SMALL_AN_TX_P =20

MACRO_BH_TX_P =23	#dbm
SMALL_BH_TX_P =23

ADJ_NODE_DIST =199 #meter for adjacent for routing


MAX_NUM_AP =5
MAX_NUM_SRC =100

CHANNEL_ALLOC_ALG_RANDOM =0
CHANNEL_ALLOC_ALG_MDF =1
CHANNEL_ALLOC_ALG_GA =2


class Simul:
    def __init__(self, space_x, space_y):
        self.xLim = space_x
        self.yLim = space_y
        self.numNodes =0
        self.numUe=0
        self.numEnb=0
        self.numMacro=0
        self.numSmall=0
        self.numLinks=0
        self.simCh=None
        self.ueDataRate = 0
        self.nodes = []
        self.enbs = []
        self.ues = []
        self.hotspotUe = []
        self.links = []
        self.bhlinks = {}
        self.anDownlinks = {}
        self.anUplinks = []
        self.nodeLinkMap = {}
        self.enbLinkMap = {}
        self.ueLinkMap = {}
        self.nxTopo = None
        self.edges = {}
        self.scellAcessLink = {}
        self.mcellAcessLink = {}
        self.blockedFlow = 0
        self.blocking = 0
        self.blockedNode = []

        # Path info
        self.currentUePath = {}
        #self.currentUeInLink = {}

        self.log = simLog(self)

    def load_data(self, args):
        self.xLim = args[0]
        self.yLim = args[1]
        self.numNodes = args[2]
        self.numUe = args[3]
        self.numEnb = args[4]
        self.numMacro = args[5]
        self.numSmall = args[6]
        self.numLinks = args[7]
        self.simCh = args[8]
        self.ueDataRate = args[9]
        self.nodes = args[10]
        self.enbs = args[11]
        self.ues = args[12]
        self.hotspotUe = args[13]
        self.links = args[14]
        self.bhlinks = args[15]
        self.anDownlinks = args[16]
        self.anUplinks = args[17]
        self.nodeLinkMap = args[18]
        self.enbLinkMap = args[19]
        self.ueLinkMap = args[20]
        self.nxTopo = args[21]
        self.edges = args[22]
        self.scellAcessLink = args[23]
        self.blockedFlow = args[24]
        self.blocking = args[25]

    def SetMeNB(self, x, y, radius):
        pEnb = eNB(MACRO_AN_TX_P, MACRO_BH_TX_P, ENB_TYPE_MACRO, radius);#AN, BN tx power
        pEnb.id = 0
        pEnb.x = x; pEnb.y = y
        pEnb.numPRB = BW20MH_PRB
        self.log.macroLoc = (x, y)
        self.enbs.append(pEnb);
        self.nodes.append(pEnb);
        self.numEnb += 1;
        self.numMacro +=1;
        self.numNodes += 1

    def SetSeNB(self, x, y, radius):
        pEnb = eNB(SMALL_AN_TX_P, SMALL_BH_TX_P, ENB_TYPE_SMALL, radius);
        self.numSmall +=1;
        self.numNodes += 1
        self.numEnb = self.numMacro + self.numSmall;
        pEnb.id = self.numEnb -1
        pEnb.x =x; pEnb.y= y;
        pEnb.numPRB = BW20MH_PRB
        self.log.smallLoc[pEnb.id] = (x, y)
        self.enbs.append(pEnb);
        self.nodes.append(pEnb);

    def generateUE(self, n, SENB_ISD, OFFSET):
        # UE100

        self.numUe = n
        self.numNodes += n
        for i in range(self.numUe):
            # index = i // 4
            #
            # x = index // 5
            # y = index % 5

            pUe = UE();

            findSC = False
            while not findSC :
                # pUe.x = random.randint(OFFSET+SENB_ISD*x, OFFSET+SENB_ISD*(x+1))
                # pUe.y = random.randint(OFFSET+SENB_ISD*y, OFFSET+SENB_ISD*(y+1))

                pUe.x = random.randint(0, self.xLim)
                pUe.y = random.randint(0, self.yLim)

                for e in self.enbs :
                    dist = getDist(pUe, e)
                    if dist < e.maxCellRadius and e.id >0:
                        findSC = True
                        break

            #print("x,y", pUe.x,pUe.y)
            pUe.id = i+ self.numEnb;
            #print("{} - {}/{}".format(pUe.id, pUe.x, pUe.y))

            self.ues.append(pUe);
            self.nodes.append(pUe);
            self.log.ueLoc[pUe.id] = (pUe.x, pUe.y)

    def generateNoHotSpotUE(self, n):
        bfind = False
        self.numUe += n
        self.numNodes += n
        for i in range(n):
            pUe = UE()
            pUe.x = random.randint(0, self.xLim)
            pUe.y = random.randint(0, self.yLim)
            while bfind == False:
                for e in self.enbs:
                    if e.id >= self.numMacro + 1 and getDist(pUe, e) <= e.maxCellRadius:
                        bfind = True
                if bfind == True:
                    pUe.x = random.randint(0, self.xLim)
                    pUe.y = random.randint(0, self.yLim)
                else:
                    break
            pUe.id = i+self.numUe + self.numEnb + 1
            self.ues.append(pUe)
            self.nodes.append(pUe)
            bfind = False;

    def generateHotSpotUE(self, n):
        bfind = False;
        for i in range(n):
            pUe = UE();
            pUe.x = random.randint(0, self.xLim) #rand() % xLim;
            pUe.y = random.randint(0, self.yLim)
            while bfind == False:
                for e in self.enbs:
                    if e.id >= self.numMacro + 1 and getDist(pUe, e) > MIN_UE_SMALL_DIST and getDist(pUe, e) <= e.maxCellRadius:
                        bfind = True
                if bfind == False:
                    pUe.x = random.randint(0, self.xLim)
                    pUe.y = random.randint(0, self.yLim)
            pUe.id = i+ self.numUe + self.numEnb + 1
            self.ues.append(pUe)
            self.nodes.append(pUe)
            bfind = False
        self.numUe += n
        self.numNodes += n


    def findNeighborEnbs(self):#find eNB with 60GHz backhaul link
        for it in self.enbs:
            for itNext in self.enbs:
                dist = getDist(it, itNext)
                if dist < it.maxBhDist and it.id != itNext.id:
                    it.addAdjNodes(itNext)
                    pLink = self.findBhLink(it, itNext)
                    if pLink == None:
                        pLink = Link(it, itNext, dist, MMWAVE_60, LINK_TYPE_BACKHAUL, MAX_NUM_ANT)
                        pLink.setChannel(MM60_RX_ANT_GAIN, MM60_TX_ANT_GAIN)
                        pLink.setLinkBw(BW_1000MH, 1)
                        pLink.setLinkTxp(it.BhTxPower)
                        self.bhlinks[(it.id, itNext.id)] = pLink
                        it.addBhLinks(pLink)	#outgoing backhaul
                        itNext.addBhLinks(pLink)

        #print("bhlinks", self.bhlinks.keys())

    def findNeighborUEs(self):  # downlink
        for it in self.enbs:
            for itNext in self.ues:
                dist = getDist(it, itNext)
                if dist < it.maxCellRadius and it.id != itNext.id:
                    it.addAdjUEs(itNext)
                    pLink = self.findAnDownLink(it, itNext)
                    if pLink == None:
                        if it.enbType == ENB_TYPE_MACRO:
                            pLink = Link(it, itNext, dist, MICWAVE_MACRO, LINK_TYPE_ACCESS, MAX_NUM_ANT);
                        elif it.enbType == ENB_TYPE_SMALL:
                            pLink = Link(it, itNext, dist, MICWAVE_SMALL, LINK_TYPE_ACCESS, MAX_NUM_ANT);
                        pLink.setChannel(AN_RX_ANT_GAIN, AN_TX_ANT_GAIN)
                        pLink.setLinkBw(BW_20MH, BW20MH_PRB)
                        pLink.setLinkTxp(it.AnTxPower);
                        self.anDownlinks[(it.id, itNext.id)] = pLink
                        if it.id == 0:  # macro cell access links
                            self.mcellAcessLink[(it.id, itNext.id)] = pLink
                            self.hotspotUe.append(itNext)
                        else :  # small cell access links
                            self.scellAcessLink[(it.id, itNext.id)] = pLink
                            self.hotspotUe.append(itNext)

                        it.addAnLinks(pLink);



    def findReachableEnb(self): #uplink is not used
    	for it in self.ues:
            tmpLinkList = []
            for itNext in self.enbs:
                dist = getDist(it, itNext)
                if dist < itNext.maxCellRadius and it.id != itNext.id:
                    it.addAdjEnbs(itNext);
                    pLink = self.findAnUpLink(it, itNext)
                    if pLink == None:
                        if itNext.enbType == ENB_TYPE_MACRO:
                            pLink = Link(it, itNext, dist, MICWAVE_MACRO, LINK_TYPE_ACCESS, MAX_NUM_ANT);
                        elif itNext.enbType == ENB_TYPE_SMALL:
                            pLink = Link(it, itNext, dist, MICWAVE_SMALL, LINK_TYPE_ACCESS, MAX_NUM_ANT);
                        pLink.setChannel(AN_RX_ANT_GAIN, AN_TX_ANT_GAIN);
                        pLink.setLinkBw(BW_20MH, BW20MH_PRB);
                        pLink.setLinkTxp(it.TxPower);
                        self.anUplinks.append(pLink);
                        it.addAnLinks(pLink);

    def checkPathCapa(self, p):
        bCapable = True
        bottleNeck = 10**10
        #print("test", p)
        for idx in range(len(p)-1):
            if (p[idx], p[idx+1]) in self.anDownlinks: #access links
                pLink = self.anDownlinks[(p[idx], p[idx+1])]
                reqRbs = math.ceil(self.ueDataRate/(pLink.prbCapa))
                #print("allocProb : {}, req : {}".format(pLink.n[0].allocPrb, reqRbs))
                #print("max : {}".format(pLink.maxPRB) )
                if pLink.n[0].allocPrb + reqRbs > pLink.maxPRB:
                    bCapable = False
                    minRate = (pLink.maxPRB - pLink.n[0].allocPrb)* pLink.prbCapa
                    #print("min Rate : {} / bottle nect : {}".format(minRate, bottleNeck))
                    if minRate < bottleNeck:
                        bottleNeck = minRate

            elif (p[idx], p[idx+1]) in self.bhlinks: #bh links
                pLink = self.bhlinks[(p[idx], p[idx+1])]
                #print("bhlink current state : {}, max : {} => {}".format(pLink.currentState, pLink.maxCapa, pLink.currentState + self.ueDataRate > pLink.maxCapa))
                if pLink.currentState + self.ueDataRate > pLink.maxCapa:
                    bCapable = False
                    minRate = pLink.maxCapa - pLink.currentState

                    #print("min Rate : {} / bottle nect : {}".format(minRate, bottleNeck))
                    if minRate < bottleNeck:
                        bottleNeck = minRate
            else :
                print("error path", p, p[idx], p[idx+1])
        #print("bottleNeck", bCapable, bottleNeck, abs(bottleNeck))
        if bottleNeck == 10**10 : bottleNeck = self.ueDataRate
        return bCapable, abs(bottleNeck)

    def placeUserFlowOnLinks(self, u, p, bFind, bRate):
        for idx in range(len(p)-1):
            if (p[idx], p[idx+1]) in self.anDownlinks: #access links
                pLink = self.anDownlinks[(p[idx], p[idx+1])]
                if bFind:
                    reqRbs = math.ceil(self.ueDataRate/(pLink.prbCapa))
                    pLink.n[0].allocPrb += reqRbs
                    u.flowRate = self.ueDataRate
                else:
                    pLink.n[0].allocPrb += math.ceil(bRate/(pLink.prbCapa))
                    u.flowRate = bRate

            elif (p[idx], p[idx+1]) in self.bhlinks: #bh links
                pLink = self.bhlinks[(p[idx], p[idx+1])]
                if bRate !=0:
                    #print("current state : {}-> {}/ f:{}".format(pLink.currentState, pLink.currentState+bRate, pLink.assNumFlows+1))
                    pLink.currentState += bRate
                    pLink.assNumFlows += 1



     #update routing and add flows to each edge
    def updateUserFlowPath(self):
        blocked = 0
        self.blockedNode = []
        #reset backhaul
        for l in self.bhlinks.values():
            l.assNumFlows = 0
            l.currentState = 0
        #reset access
        for n in self.enbs:
            n.allocPrb = 0
        for u in self.ues:
            u.flowRate = 0

            path = self.nxTopo.getShortestPath(0, u.id)
            # print("{} - path : {}".format(u.id, path))
            if len(path) == 2 :
                bFind, bRate = self.checkPathCapa([0, u.id]) #marcro AN
                pLink = self.anDownlinks[(0, u.id)]
                reqRbs = math.ceil(self.ueDataRate/(pLink.prbCapa))
                if bFind:
                    pLink.n[0].allocPrb += reqRbs
                    u.flowRate = self.ueDataRate
                else:
                    pLink.n[0].allocPrb += math.ceil(bRate/(pLink.prbCapa))
                    u.flowRate = bRate

                    if bRate == 0 :
                        blocked +=1
                        self.blockedNode.append(u.id)
                    self.log.MSG_LOG(LOG_TYPE_UE, 'UE '+str(u.id) + ' blocked')
            else: #via scell backhaul nework
                bFind, bRate = self.checkPathCapa(path)
                self.placeUserFlowOnLinks(u, path, bFind, bRate)

                if bRate == 0:
                    blocked += 1
                    self.blockedNode.append(u.id)

            self.currentUePath[u.id] = (path, bFind)

        # self.currentUeInLink = {}
        # for k, l in self.bhlinks.items():
        #     self.currentUeInLink[k] = l.currentState

        self.blocking = blocked / len(self.ues)

    def genNxTopo(self):
        self.edges = {**self.bhlinks, **self.scellAcessLink, **self.mcellAcessLink}
        topo_edges = {**self.bhlinks, **self.scellAcessLink}
        self.nxTopo = Topology(self.numNodes, self.numSmall, list(topo_edges.keys()))


    def setPossible_mcellU(self):
        for link in self.mcellAcessLink.values() :
            reqRbs = math.ceil(self.ueDataRate / (link.prbCapa))
            self.nxTopo.updatePossible_mcellU(int(link.maxPRB/reqRbs))

            return

    def updateTopo(self, edges):
        self.nxTopo.updateTopoWithEdge(edges)

    def findBhLink(self, n, m):
    	for l in self.bhlinks.values():
    		pNode = l.n
    		if pNode[0].id == n.id and pNode[1].id == m.id:
    			return l
    	return None

    def findExistDLink(self, u):
        for k, v in self.anDownlinks.items():
            if v.n[1] == u:
                return k, v
        return None, None

    def findAnDownLink(self, n, m): #eNB* n, UE* m
        for l in self.anDownlinks.values():
            pNode = l.n
            if pNode[0].id == n.id and pNode[1].id == m.id:
                return l;
        return None;

    def findAnUpLink(self, n, m): #UE* n, eNB* m
    	for l in self.anUplinks:
    		pNode = l.n
    		if pNode[0].id == n.id and pNode[1].id == m.id:
    			return l;
    	return None

    def findDLInfLinks(self): #downlink
    	for u in self.ues:
    		for e in u.adjEnbs:	#candiate eNBs
    			l = self.findAnDownLink(e, u);
    			for a in e.anLinks: #anlinks of adj eNBs
    				pNode = a.n;
    				if pNode[1].id != u.id: #do not add own link
    					l.addinfLinks(a);

    def findBhInfLinks(self):
    	for u in self.enbs:
            for e in u.adjEnbs:				#adj eNBs
                l = self.findBhLink(u, e)
                for a in u.bhLinks:
                    if a != l:
                        l.addinfLinks(a);
                for a in e.bhLinks:
                    if a != l:
                        l.addinfLinks(a);

    def calLinkCapacity(self):
    	for lit in self.bhlinks.values():
            lit.calBhLinkCapa() #calSingleBhLinkCapacity();
            self.log.bh[(lit.n[0].id, lit.n[1].id)] = lit.maxCapa
    	for i, lit in self.anDownlinks.items():
            rbRate = lit.calAnLinkRBCapacity();
            lit.maxCapa = lit.maxPRB * rbRate

            print("test", i, lit.maxCapa)
            self.log.an[(lit.n[0].id, lit.n[1].id)] = lit.maxCapa
    	for lit in self.anUplinks: #not used
    		lit.calSingleAnLinkCapacity();

    def getLinksLoad(self):
        linkWorkload = []
        # print("t bh", self.bhlinks.keys())
        # print("t scell", self.scellAcessLink.keys())
        # print("============================================")
        for b in self.bhlinks.values():
            linkWorkload.append(b.currentState/b.maxCapa)

        for n in self.enbs:
            if len(n.anLinks) != 0 :
                linkWorkload.append(n.allocPrb/n.anLinks[0].maxPRB)
        #padding for access links = UE * 2 (each for macro, scell)
        #padding = [ 0 for i in range(self.numUe * 2 - len(self.anDownlinks.values()))]
        #return linkWorkload + padding
        return linkWorkload


    def calEnergyConsumption(self):
        totalPower = 0
        #print("calEnergyConsumption")
        for n in self.enbs:
            if n.id == 0: #macro6
                if sum([bh.assNumFlows for bh in n.bhLinks]) ==0 and n.allocPrb ==0:
                    macroTotalPower = 0
                    eNBPower = 0
                else:
                    macroAnPower = 130 + 4.7* (n.allocPrb/n.numPRB* 20) #watt
                    macroBhPower = 3.9
                    for bh in n.bhLinks:
                        if bh.n[0].id == n.id :
                            macroBhPower += 0.224*(bh.currentState/bh.maxCapa)

                    eNBPower = macroAnPower + macroBhPower
            else: #senbs
                if sum([bh.assNumFlows for bh in n.bhLinks]) ==0 and n.allocPrb ==0:
                    scellTotalPower = 0
                    eNBPower = 0
                else:
                    scellAnPower = 6.8 + 4.0* (n.allocPrb/n.numPRB *0.13) #watt
                    scellBhPower = 3.9
                    for bh in n.bhLinks:
                        if  bh.n[0].id == n.id :
                            scellBhPower += 0.224* min((bh.currentState/bh.maxCapa), 1)

                    eNBPower = scellAnPower + scellBhPower

                #print("test", n.id, eNBPower)
            totalPower += eNBPower
        #print("sum", totalPower)
        return totalPower

    def getEnergyConsumption(self):
        BhPower = 0
        AlPower = 0
        for n in self.enbs:
            if n.id == 0: #macro6
                if sum([bh.assNumFlows for bh in n.bhLinks]) == 0 and n.allocPrb ==0:
                    continue
                else:
                    AlPower += (130 + 4.7 * (n.allocPrb/n.numPRB* 20)) #watt
                    macroBhPower = 3.9
                    for bh in n.bhLinks:
                        if bh.n[0].id == n.id:
                            macroBhPower += 0.224 * (bh.currentState / bh.maxCapa)
                    BhPower += macroBhPower
            else: #senbs
                if sum([bh.assNumFlows for bh in n.bhLinks]) ==0 and n.allocPrb ==0:
                    continue
                else:
                    AlPower += (6.8 + 4.0* (n.allocPrb/n.numPRB *0.13)) #watt
                    scellBhPower = 3.9
                    for bh in n.bhLinks:
                        if bh.n[0].id == n.id:
                            scellBhPower += 0.224 * min((bh.currentState / bh.maxCapa), 1)
                    BhPower += scellBhPower

        return AlPower, BhPower, 1- ((AlPower + BhPower) /self.calMaxEnergy())

    def calMaxEnergy(self):
        totalPower = 0
        for n in self.enbs:
            if n.id == 0: #macro
                macroAnPower = 130 + 4.7* 20 #watt
                macroBhPower = 3.9
				#print("macro", len(n.bhLinks))
                for bh in n.bhLinks:
                    if bh.n[0].id == n.id:
                        macroBhPower += 0.224
                eNBPower = macroAnPower + macroBhPower
            else: #senbs
                scellAnPower = 6.8 + 4.0*0.13 #watt
                scellBhPower = 3.9
				#print("sm", len(n.bhLinks))
                for bh in n.bhLinks:
                    if bh.n[0].id == n.id:
                        scellBhPower += 0.224
                eNBPower = scellAnPower + scellBhPower
            totalPower += eNBPower
        return totalPower


    def getActiveLink(self):
        usedLink = 0
        for k, l in self.bhlinks.items():
            #print("{} : {}".format(k, l.currentState))
            if l.currentState != 0 : usedLink +=1

        return usedLink

    def getActiveSenbs(self):
        usedSenbs = 0
        for n in self.enbs:
            if n.id != 0:
                #print("{} : {}".format(n.id, sum([bh.currentState for bh in n.bhLinks])))
                if sum([bh.currentState for bh in n.bhLinks]) != 0 :
                    usedSenbs +=1

        return usedSenbs

    def getLinkCapa(self):
        LinkCapa = {}
        for k, l in self.bhlinks.items() :
            LinkCapa[k] = l.maxCapa / self.ueDataRate

        return LinkCapa

    def getUePath(self):
        return self.currentUePath

    def getUeInLink(self):
        temp = {}
        for i, l  in self.bhlinks.items() :
            temp[i] = l.currentState

        return temp

    def getHospotUE(self):
        SCAL = {}
        for i in set(self.hotspotUe):
            c = self.hotspotUe.count(i)
            SCAL[c] = SCAL[c]+1 if c in SCAL else 1

        return sorted(SCAL.items())

    def getNotHospotUE(self):
        NHUE = 0
        for ue in self.ues :
            if not ue in self.hotspotUe :
                NHUE +=1

        return NHUE

    def getScellAcessLink(self):
        SCAL = {}
        for k in self.scellAcessLink.keys() :
            if k[0] in SCAL :
                SCAL[k[0]] +=1
            else :
                SCAL[k[0]] = 1

        return SCAL
