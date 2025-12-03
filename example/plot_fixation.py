import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ast


w, C =0.4,1.
dynamics= "DBB"
omega,L = None, None
Q=2
V_vector=[1.,2.,5.,10.]
M=3
game_vector=["CPD"]


topology="circle"




#------------------------------First set of Plots------------------------------



fig_merge = plt.figure(figsize=(15, 7))
gs_merge = gridspec.GridSpec(ncols=2, nrows=1)#, figure=figC)
ax_vector_merge = [None]*(2*len(game_vector))

for j in range(len(game_vector)):

	game = game_vector[j]

	ax_vector_merge[j*2]=plt.subplot(gs_merge[j,0])
	ax_vector_merge[j*2].plot([0.01,100],[1/float(M*Q)]*2,"--",label="Neutral",color='grey')

	for k in range(len(V_vector)):


		file_object = open("data/{}_{}_fixation_h_game={}_M={}_Q={}_w={}_V={}_C={}_omega={}_L={}.txt".format(topology,dynamics,game,M,Q,w,V_vector[k],C,omega,L), "r")

		
		all_lines=file_object.readlines()
		h_vector=ast.literal_eval(all_lines[0])


		for line in all_lines[1:]:
			data= line.split(":")
			plt.plot(h_vector,np.array(ast.literal_eval(data[0].strip()))[:,0,0],label="V={}".format(V_vector[k]), linestyle='solid', marker='o', markersize=6)
				


	ax_vector_merge[j*2].set_xscale("log")
	ax_vector_merge[j*2].set_xlabel("h")
	ax_vector_merge[j*2].set_ylabel("fixation probability")
	ax_vector_merge[j*2].set_title("F.P. of Cooperators under {}".format(game))




	ax_vector_merge[j*2+1]=plt.subplot(gs_merge[j,1])
	ax_vector_merge[j*2+1].plot([0.01,100],[1/float(M*Q)]*2,"--",label="Neutral",color='grey')



	for k in range(len(V_vector)):



		file_object = open("data/{}_{}_fixation_h_game={}_M={}_Q={}_w={}_V={}_C={}_omega={}_L={}.txt".format(topology,dynamics,game,M,Q,w,V_vector[k],C,omega,L), "r")
		
		all_lines=file_object.readlines()
		h_vector=ast.literal_eval(all_lines[0])

		for line in all_lines[1:]:
			data= line.split(":")
			plt.plot(h_vector,np.array(ast.literal_eval(data[0].strip()))[:,0,1],label="V={}".format(V_vector[k]), linestyle='solid', marker='o', markersize=6)
			

	ax_vector_merge[j*2+1].set_xscale("log")
	ax_vector_merge[j*2+1].set_xlabel("h")
	ax_vector_merge[j*2+1].set_ylabel("fixation probability")
	ax_vector_merge[j*2+1].set_title("F.P. of Defectors under {}".format(game))

	


lines, labels = ax_vector_merge[0].get_legend_handles_labels()
fig_merge.legend(lines,labels, loc='center right',fontsize='xx-large')
fig_merge.savefig("plots/4Vs_{}_{}_fixation_h_5games_M={}_Q={}_w={}_C={}_omega={}_L={}.pdf".format(topology,dynamics,M,Q,w,C,omega,L))
fig_merge.show()

