# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:16:40 2021

@author: Gachon
"""
import networkx as nx
import numpy as np; #NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt #for plotting
from senb import SeNB
from menb import MeNB

def genPPPTopo(lambda_senb=10, lambda_ue=10):
    xMin=0;xMax=1;
    yMin=0;yMax=1;
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta;
    
    #Simulate a Poisson point process
    numSenb = np.random.poisson(lambda_senb*areaTotal);#Poisson number of points
    senb_x = xDelta*np.random.uniform(0,1,numSenb)+xMin;#x coordinates of Poisson points
    senb_y = yDelta*np.random.uniform(0,1,numSenb)+yMin;#y coordinates of Poisson points
    senbs = [SeNB(senb_x[i], senb_y[i]) for i in range(numSenb)]
    
    menb = MeNB(0.5, 0.5); #one macro enb
    senb_x = np.append(senb_x, [0.5]);
    senb_y = np.append(senb_y, [0.5]);
    p = {i: (senb_x[i], senb_y[i]) for i in range(numSenb+1)}
    G = nx.random_geometric_graph(numSenb+1, 0.4, pos=p) #radius =0.2
    nx.draw(G)
    
    numUe = np.random.poisson(lambda_ue*areaTotal);#Poisson number of points
    ue_x = xDelta*np.random.uniform(0,1,numUe)+xMin;#x coordinates of Poisson points
    ue_y = yDelta*np.random.uniform(0,1,numUe)+yMin;#y coordinates of Poisson points
    senbs = [SeNB(ue_x[i], ue_y[i]) for i in range(numUe)]

# def getGridTopo():
#     netTopo->SetMeNB(0, 250, 1000);	//macro 1 km
#
# 	netTopo->SetSeNB(50, 250, 80, 1);
# 	netTopo->SetSeNB(100, 250, 80, 2);
# 	netTopo->SetSeNB(150, 250, 80, 3);
# 	netTopo->SetSeNB(200, 250, 80, 4);
# 	netTopo->SetSeNB(250, 250, 80, 5);
#
# 	netTopo->SetSeNB(0, 200, 80, 6);	//senb 200m
# 	netTopo->SetSeNB(50, 200, 80, 7);
# 	netTopo->SetSeNB(100, 200, 80, 8);
# 	netTopo->SetSeNB(150, 200, 80, 9);
# 	netTopo->SetSeNB(200, 200, 80, 10);
# 	netTopo->SetSeNB(250, 200, 80, 11);
#
# 	netTopo->SetSeNB(0, 150, 80, 12);	//senb 200m
# 	netTopo->SetSeNB(50, 150, 80, 13);
# 	netTopo->SetSeNB(100, 150, 80, 14);
# 	netTopo->SetSeNB(150, 150, 80, 15);
# 	netTopo->SetSeNB(200, 150, 80, 16);
# 	netTopo->SetSeNB(250, 150, 80, 17);
#
# 	netTopo->SetSeNB(0, 100, 80, 18);	//senb 200m
# 	netTopo->SetSeNB(50, 100, 80, 19);
# 	netTopo->SetSeNB(100, 100, 80, 20);
# 	netTopo->SetSeNB(150, 100, 80, 21);
# 	netTopo->SetSeNB(200, 100, 80, 22);
# 	netTopo->SetSeNB(250, 100, 80, 23);
#
# 	netTopo->SetSeNB(0, 50, 80, 24);	//senb 200m
# 	netTopo->SetSeNB(50, 50, 80, 25);
# 	netTopo->SetSeNB(100, 50, 80, 26);
# 	netTopo->SetSeNB(150, 50, 80, 27);
# 	netTopo->SetSeNB(200, 50, 80, 28);
# 	netTopo->SetSeNB(250, 50, 80, 29);
#
# 	netTopo->SetSeNB(0, 0, 80, 30);	//senb 200m
# 	netTopo->SetSeNB(50, 0, 80, 31);
# 	netTopo->SetSeNB(100, 0, 80, 32);
# 	netTopo->SetSeNB(150, 0, 80, 33);
# 	netTopo->SetSeNB(200, 0, 80, 34);
# 	netTopo->SetSeNB(250, 0, 80, 35);
