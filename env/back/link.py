# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:57:46 2021

@author: Gachon
"""
from env.hetNet.channel import *
from env.hetNet.util import *
B_PRB =180000
SNR_SHANON_GAP =7	#7db
MAX_SNR =27
MIN_SNR =-10

LINK_TYPE_ACCESS = 0
LINK_TYPE_BACKHAUL = 1

class Link:
    def __init__(self, _x, _y, _len, s, _type, Ant):
        self.n = (_x, _y)
        self.length = _len; 
        self.linkType = _type;
        self.numAnt = Ant;
        self.ch = Channel(s)
        self.bandwidth = 0
        self.maxPRB = 0
        self.assignPRB = 0
        self.noise = 0
        self.sinr = 0
        self.snr = 0
        self.capa = 0
        self.maxCapa = 0
        self.singleCapa = 0
        self.prbCapa = 0
        self.txp =0
        self.power = 0
        self.assNumFlows = 0
        self.infLinks = set()

        self.activeLink = 0 # assNumFlows, to be corrected later

    def addinfLinks(self, l): #Link* l
        self.infLinks.add(l)
    
    def setLinkBw(self, b, nPrbs):
    	self.bandwidth = b; 
    	self.noise = self.ch.noise + Watt_dBw(self.bandwidth);	#dBm
    	self.maxPRB = nPrbs;
    
    def setChannel(self, Gr, Gt):
        self.ch.setChParam(Gr, Gt)
    
    def setLinkTxp(self, tp):
        self.txp = tp

    #receive power, dBm
		#SNR(dB) + Nth(dBm) + NF(dB) + PL(dB) + TX_loss(dB) + RX_loss(dB) - Gtx(dBi) - Grx(dBi) + Link_margin
		#Nth(dBm) + NF(dB) + PL(dB) + TX_loss(dB) + RX_loss(dB) -Gtx(dBi) - Grx(dBi) + Link_margin
    def getLinkRcvPower(self):
        RxP = 0;
        if self.linkType == LINK_TYPE_ACCESS:
            RxP = self.txp - self.ch.getPathLoss(self.length*0.001) + self.ch.RxGain + self.ch.TxGain - self.ch.il - self.ch.NFig - self.ch.cbLoss - self.ch.getGasLoss(self.length) - self.ch.getRainLoss(self.length);
        elif self.linkType ==  LINK_TYPE_BACKHAUL:
            RxP = self.txp - self.ch.getPathLoss(self.length*0.001) - self.ch.NFig + self.ch.RxGain + self.ch.TxGain - self.ch.TxLoss - self.ch.RxLoss - self.ch.getGasLoss(self.length) - self.ch.getRainLoss(self.length)- LINK_MARGIN;
        return RxP;

    def getSNR(self, rxP):
    	self.snr = rxP - self.noise;
    	return self.snr

    def calAnLinkRBCapacity(self):
        self.snr = self.getSNR(self.getLinkRcvPower())
        self.snr -= self.ch.getSlowFading()
        self.prbCapa = self.numAnt * B_PRB * self.ch.getChannelBph(self.snr)
        return self.prbCapa

    def calSingleBhLinkCapacity(self):
        self.snr = self.getSNR(self.getLinkRcvPower());
        self.maxCapa = self.bandwidth * self.ch.getChannelBph(self.snr)
        return self.maxCapa
    
    def calBhLinkCapa(self):
        self.maxCapa = self.calSingleBhLinkCapacity()/len(self.infLinks) #TDM
        
    def calAnLinkCapa(self):
        self.calSingleAnLinkCapacity()
        self.maxCapa = self.numAnt * self.bandwidth * self.ch.getChannelBph(self.snr);
        self.capa = self.assignPRB * self.prbCapa; #tbd

    def getLinkSinr(self):
    	sumInfWattPower = 0; sinr = 0;
    	for l in self.infLinks:
    		sumInfWattPower += dBm_Watt(l.getLinkRcvPower());
    	sinr = dBm_Watt(self.getLinkRcvPower()) / (sumInfWattPower + dBm_Watt(self.noise));
    	return sinr;
    
    def getLinkEnergy(self, t):
    	txp = self.n[0].getTxPower(n[1]);
    	energyWatt = dBm_Watt(txp) / (self.maxPRB * self.numAnt);
    	return energyWatt /(B_PRB * math.log2(1 + dB_Watt(self.sinr)));


    def getPRBLinkEnergy(self, t):
    	self.sinr = self.getSNR(self.getLinkRcvPower());
    	if self.sinr > MAX_SNR:
    		self.sinr = MAX_SNR;
    	elif self.sinr < MIN_SNR:
    		self.sinr = MIN_SNR;
            
    	if t== LINK_TYPE_ACCESS:
    		energy = dBm_Watt(txp)/(self.maxPRB * self.numAnt) * 1/(self.B_PRB * log2(1 + dB_Watt(self.sinr)));
    	elif t== LINK_TYPE_BACKHAUL:
    		eDbm = self.sinr + self.ch.getPathLoss(self.length*0.001) + self.noise + self.ch.getNFigure() \
            - self.ch.getAntRxGain() - self.ch.getAntTxGain() + self.ch.getTxLoss() + self.ch.getRxLoss() \
            + self.ch.getGasLoss(self.length) + self.ch.getRainLoss(self.length) + LINK_MARGIN;
    		energy = dBm_Watt(eDbm);
    	return energy;