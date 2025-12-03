import TRMbeta
import numpy as np

#Example of building a population under the territorial raider model and extracting the fixation probabilities

#Population parameters:
#	M 				: number of places in network
#	h 				: home fidelity
#	w 				: intensity of selection
#	C 				: cost (payoff)
#	V 				: value (payoff)
#	adjacency_matrix: of the network of places
#	game 			: can be "PD", "PDV", "CPD", "SH", "FSH", "VD", "TVD", "SD", "TSD", "HD"
#	dynamics		: can be "BDB", "BDD", "DBD",  "DBB", "LB", "LD"
#	Q 				: size of subpopulations
#	omega 			: power base parameter used under the "PDV" game
#	L 				: threshold parameter used under the "SH", "FSH", "TVD", "TSD" games
#	topology		: topology of spatial network

#Builds the adjacency matrix
M=3
adjacency_matrix = np.zeros((M, M), dtype=int)
for i in range(M):
    adjacency_matrix[i, (i-1) % M] = 1
    adjacency_matrix[i, (i+1) % M] = 1


#Builds the population
pop = TRMbeta.Population(M=M, h=10., w=0.3, C=1., V=5., game="CPD", dynamics="DBB", adjacency_matrix=adjacency_matrix, Q=2, topology="circle")

#Provides both the aritmetic and the temperature-weighted average of the probability of one single mutant using C fixating 
#on a population of D and vice-versa. Result in the form of [[fix_C, fix_D],[fixT_C, fixT_D]]
fix_prob = pop.get_fixation_probabilities()
print(fix_prob)

#Calculates the fixation probabilities for different valuess of h between 0.01 and 100 and saves them in a data folder
TRMbeta.get_fixation_h(M=M, w=0.4, C=1., V=1., game="CPD", dynamics="DBB", adjacency_matrix=adjacency_matrix, Q=2, topology="circle")
TRMbeta.get_fixation_h(M=M, w=0.4, C=1., V=2., game="CPD", dynamics="DBB", adjacency_matrix=adjacency_matrix, Q=2, topology="circle")
TRMbeta.get_fixation_h(M=M, w=0.4, C=1., V=5., game="CPD", dynamics="DBB", adjacency_matrix=adjacency_matrix, Q=2, topology="circle")
TRMbeta.get_fixation_h(M=M, w=0.4, C=1., V=10., game="CPD", dynamics="DBB", adjacency_matrix=adjacency_matrix, Q=2, topology="circle")
