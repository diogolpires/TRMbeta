import numpy as np
import matplotlib.pyplot as plt
from TRMbeta import *
#import ast
import os

	
def get_fixation_h(M, w, C, V, game, dynamics, adjacency_matrix, Q, omega=None, L=None, topology=None):
	"""Calculates the fixation probabilities for a set of values of home fidelity between 0.01 and 100 and saves them in a .txt file."""


	file_object = open("data/{}_{}_fixation_h_game={}_M={}_Q={}_w={}_V={}_C={}_omega={}_L={}.txt".format(topology,dynamics,game,M,Q,w,V,C,omega,L), "w") #CHANGE FROM "w" TO "a", in order to append, instead of overwrite


	h_min=0.01
	h_max=100.
	#n_points=50
	n_points=25
	h_vector=[h_min*(h_max/h_min)**(float(i)/float(n_points)) for i in range(n_points+1)]
	file_object.write(str(h_vector)+ "\n")


	#v_vector=[2.,1.,0.5,0.25,0.05]
	#v_vector=[2.,1.,0.5,0.05]
	#v_vector=[2.]


	#file_object.write("v={}: [".format(v))
	
	file_object.write("[")
	file_object.flush()
	for h in h_vector[:-1]:
		pop = Population(M, h, w, C, V, game, dynamics, adjacency_matrix, Q, omega, L, topology)
		fix_prob = pop.get_fixation_probabilities()
		#fix_vector.append(fix_prob)
		#print("Fixation Probabilities (C,D): ",fix_prob, "  Neutral Fixation: {}".format(1/float(N)))
		file_object.write("{},".format(fix_prob))
		file_object.flush()
	#file_object.seek(-1,os.SEEK_END)
	for h in [h_vector[-1]]:
		pop = Population(M, h, w, C, V, game, dynamics, adjacency_matrix, Q, omega, L, topology)
		fix_prob = pop.get_fixation_probabilities()
		#fix_vector.append(fix_prob)
		#print("Fixation Probabilities (C,D): ",fix_prob, "  Neutral Fixation: {}".format(1/float(N)))
		file_object.write("{}]\n".format(fix_prob))
		file_object.flush()
	file_object.close()



