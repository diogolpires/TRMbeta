# TRMbeta

### Purpose

This package offers tools to explore evolutionary dynamics under the Territorial Raider Model. This framework allows the study of structured multiplayer interactions under social dilemmas. See a usage of this package in arXviv: For more foundational work on it see https://doi.org/10.1016/j.jtbi.2017.06.034 and https://doi.org/10.1016/j.jtbi.2012.02.025.


### Installation

Run the command:
`pip install git+https://github.com/diogolpires/TRM.git`


### Usage

A quick example displaying how to import and use the package:
````python
import TRMbeta

M=4
adjacency_matrix = np.zeros((M, M), dtype=int)
for i in range(M):
    adjacency_matrix[i, (i-1) % M] = 1
    adjacency_matrix[i, (i+1) % M] = 1


#Build the population
pop = TRMbeta.Population(M=M, h=10., w=0.3, C=1., V=5., game="CPD", dynamics="DBB", adjacency_matrix=adjacency_matrix, Q=2)

#Provides both the aritmetic and the temperature-weighted average of the probability of one single mutant using C fixating 
#on a population of D and vice-versa. Result in the form of [[fix_C, fix_D],[fixT_C, fixT_D]]
fix_prob = pop.get_fixation_probabilities()
print(fix_prob)
````