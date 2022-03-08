# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:40:51 2021

@author: Gachon
"""
import math
MM60_SPECT =60000000000
MM80_SPECT =80000000000

CABLE_LOSS =3
IMP_LOSS =5
NOISE_FIGURE =9

THERM_NOISE =-174
LINK_MARGIN =15
RX_SENS =-107.5

NORM_TX_LOSS =5
NORM_RX_LOSS =5

MACRO_GAS_LOSS =3
MICRO_GAS_LOSS =7
MM60_GAS_LOSS =10

MACRO_RAIN_LOSS =3.5
MICRO_RAIN_LOSS =8
MM60_RAIN_LOSS =12

MACRO_HEAVY_RAIN_LOSS =5
MICRO_HEAVY_RAIN_LOSS =12
MM60_HEAVY_RAIN_LOSS =18

MM60_GAS_LOSS_200M =3
MM60_GAS_LOSS_500M =7
MM60_GAS_LOSS_700M =10
MM60_GAS_LOSS_1000M =18
MM60_GAS_LOSS_1400M =20

MM60_RAIN_LOSS_200M =3.5
MM60_RAIN_LOSS_500M =8
MM60_RAIN_LOSS_700M =12
MM60_RAIN_LOSS_1000M =18
MM60_RAIN_LOSS_1400M =25

MM60_HEAVYRAIN_LOSS_200M =5
MM60_HEAVYRAIN_LOSS_500M =12
MM60_HEAVYRAIN_LOSS_700M =18
MM60_HEAVYRAIN_LOSS_1000M =24
MM60_HEAVYRAIN_LOSS_1400M =33

LTE_QPSK_13 =-0.75
LTE_QPSK_12 =1.5
LTE_QPSK_23 =3.5
LTE_QAM16_12 =7
LTE_QAM16_23 =9.5
LTE_QAM16_45 =11.5
LTE_QAM64_23 =14.7
LTE_QAM64_34 =16

LTE_QAM256_23 =19.2
LTE_MAX_MCS_SNR =23
LTE_MIN_MCS_SNR =-10

LTE_BPSK_BPH =0.1
LTE_QPSK_13_BPH =0.71
LTE_QPSK_12_BPH =0.95
LTE_QPSK_23_BPH =1.5
LTE_QAM16_12_BPH =1.9
LTE_QAM16_23_BPH =2.26
LTE_QAM16_45_BPH =2.4
LTE_QAM64_23_BPH =4
LTE_QAM64_34_BPH =4.4
LTE_QAM256_23 =19.2

MM60_BPSK =8.5
MM60_QPSK =11.5
MM60_QAM16 =14.5
MM60_QAM64 =21
MM60_QAM256 =27
MM60_MAX_MCS_SNR =27
MM60_MIN_MCS_SNR =-10

MM60_BPSK_BPH =0.19
MM60_QPSK_BPH =1.6	
MM60_QAM16_BPH =3.2	
MM60_QAM64_BPH =4.8	
MM60_QAM256_BPH =6.4

MICWAVE_MACRO = 0;
MICWAVE_SMALL = 1;
MMWAVE_60 = 2;
MMWAVE_70 = 3;

class Channel:
    def __init__(self, chType = MICWAVE_MACRO):
    	self.totCh = 0;	#num of bands
    	self.pathLoss =0;
    	self.cbLoss =0;
    	self.gasLoss =0;
    	self.rainLoss =0;
    	self.slowFading =0;
    	self.ch = chType;    
    	self.RxGain =0;
    	self.TxGain =0;
    	self.NFig =0;
    	self.il =0;
    	self.noise =0;
    	self.RxLoss =0;
    	self.TxLoss =0;
	
    def incTotCh(self):
        self.totCh += 1
        
    def setChParam(self, GRx, GTx):
        self.RxGain = GRx; self.TxGain = GTx; self.NFig = NOISE_FIGURE; 
        self.il = IMP_LOSS; self.noise = THERM_NOISE; self.cbLoss = CABLE_LOSS;
        self.RxLoss = NORM_RX_LOSS; self.TxLoss = NORM_TX_LOSS;
        if self.ch == MICWAVE_MACRO:
            self.gasLoss = MACRO_GAS_LOSS;
            self.rainLoss = MACRO_RAIN_LOSS;
        elif self.ch == MICWAVE_SMALL:
            self.gasLoss = MICRO_GAS_LOSS;
            self.rainLoss = MICRO_RAIN_LOSS;
        elif self.ch == MMWAVE_60:
            self.gasLoss = MM60_GAS_LOSS;
            self.rainLoss = MM60_RAIN_LOSS;

    def getGasLoss(self, dist):
    	if self.ch == MICWAVE_MACRO or self.ch == MICWAVE_SMALL:
    		return self.gasLoss;
    	elif self.ch == MMWAVE_60:
    		if dist < 200:
    			return MM60_GAS_LOSS_200M;
    		elif dist < 500:
    			return MM60_GAS_LOSS_500M;
    		elif dist < 700:
    			return MM60_GAS_LOSS_700M;
    		elif dist < 1000:
    			return MM60_GAS_LOSS_1000M;
    		else:
    			return MM60_GAS_LOSS_1400M;

    def getRainLoss(self, dist):
    	if self.ch == MICWAVE_MACRO or self.ch == MICWAVE_SMALL:
    		return self.gasLoss;
    	if self.ch == MMWAVE_60:
    		if dist < 200:
    			return MM60_RAIN_LOSS_200M;
    		elif dist < 500:
    			return MM60_RAIN_LOSS_500M;
    		elif dist < 700:
    			return MM60_RAIN_LOSS_700M;
    		elif dist < 1000:
    			return MM60_RAIN_LOSS_1000M;
    		else:
    			return MM60_RAIN_LOSS_1400M;


    def getPathLoss(self, dist):
        if dist == 0:
            dist = 1
        if self.ch == MICWAVE_MACRO:
            pathLoss = 128.1 + 37.6* math.log10(dist)
        elif self.ch == MICWAVE_SMALL:
            pathLoss = 140.7 + 36.7* math.log10(dist);
        elif self.ch == MMWAVE_60:
            pathLoss = 20 * math.log10(MM60_SPECT*0.000001) + 32.45 + 20 * math.log10(dist);
            #pathLoss = 20 * log10(4 * 3.14 * dist * 1000 / 0.005); 
        return pathLoss; 

    def getSlowFading(self):
        fade = 0
        if self.ch == MICWAVE_MACRO:
            fade = 6;
        elif self.ch == MICWAVE_SMALL:
            fade = 10
        elif self.ch == MMWAVE_60:
            fade = 0
        return fade;

    def getChannelBph(self, snr):
    	bph = 0;
    	if self.ch == MICWAVE_MACRO or self.ch == MICWAVE_SMALL:
    		if snr < LTE_MIN_MCS_SNR:
    			bph = 0;
    		elif snr < LTE_QPSK_13:
    			bph = LTE_BPSK_BPH;
    		elif snr < LTE_QPSK_12:
    			bph = LTE_QPSK_13_BPH;
    		elif snr < LTE_QPSK_23:
    			bph = LTE_QPSK_12_BPH;
    		elif snr < LTE_QAM16_12:
    			bph = LTE_QPSK_23_BPH;
    		elif snr < LTE_QAM16_23:
    			bph = LTE_QAM16_12_BPH;
    		elif snr < LTE_QAM16_45:
    			bph = LTE_QAM16_23_BPH;
    		elif snr < LTE_QAM64_23:
    			bph = LTE_QAM16_45_BPH;
    		elif snr < LTE_QAM64_34:
    			bph = LTE_QAM64_23_BPH;
    		else:
    			bph = LTE_QAM64_34_BPH;
    	elif self.ch == MMWAVE_60:
    		if snr < MM60_BPSK:
    			bph = 0;
    		elif snr < MM60_QPSK:
    			bph = MM60_BPSK_BPH;
    		elif snr < MM60_QAM16:
    			bph = MM60_QPSK_BPH;
    		elif snr < MM60_QAM64:
    			bph = MM60_QAM16_BPH;
    		elif snr < MM60_QAM256:
    			bph = MM60_QAM64_BPH;
    		else:
    			bph = MM60_QAM256_BPH;
                
    	return bph;