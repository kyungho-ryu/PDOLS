# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:10:48 2021

@author: Gachon
"""
import math
def getDist(n, m):
	sqX = math.pow(n.x - m.x, 2);
	sqY = math.pow(n.y - m.y, 2);
	return math.sqrt(sqX + sqY);

def dBm_Watt(d):
	return math.pow(10, (d - 30) / 10);

def dB_Watt(d):
	return math.pow(10, (d) / 10);

def dBm_mWatt(d):
	return dBm_Watt(d)/1000;

def Watt_dBw(w):
	return 10 * math.log10(w);

def Watt_dBm(w):
	return 10 * math.log10(w) + 30;
