import numpy as np
from scipy.special import comb # in Python 3
#from scipy.misc import comb # in Python 2
import matplotlib.pyplot as plt
#import time as time

class Population(object):
	"""This class allows the definition of a Population evolving under different settings.

	Attributes:
		M 				: number of places in spatial network
		h 				: home fidelity
		w 				: intensity of selection
		C 				: cost (payoff)
		V 				: value (payoff)
		adjacency_matrix: of the network of places
		game 			: can be "PD", "PDV", "CPD", "SH", "FSH", "VD", "TVD", "SD", "TSD", "HD"
		dynamics		: can be "BDB", "BDD", "DBD",  "DBB", "LB", "LD"
		Q 				: size of subpopulations
		omega 			: power base parameter used under the "PDV" game
		L 				: threshold parameter used under the "SH", "FSH", "TVD", "TSD" games
		topology		: topology of spatial network (only used for simplifying calculations)
	"""
	
	def __init__(self, M, h, w, C, V, game, dynamics, adjacency_matrix, Q ,omega=None, L=None, topology=None):

		self.M,self.h,self.w,self.C,self.V,self.Q,self.omega,self.L=  M,h,w,C,V,Q,omega,L
		self.dynamics,self.game  = dynamics,game
		
		self.adjacency_matrix, self.topology = adjacency_matrix, topology

		self.N = Q*M

		self.get_reward=getattr(self,"get_reward_"+game)
		self.get_replacement_probability=getattr(self,"get_replacement_probability_"+dynamics)

		self.generate_p()
		#print(self.p)
		self.generate_replacement_weights()
		#print(self.replacement_weights)
		#print(self.self_replacement_weights)
		self.generate_fitness()
		#print(self.fitness)
		#print("Object created.")

		#print("\n------------------------------------------------------------------------------------")
		#print("Created population with N={}, h={}, R={}, C={}, V={}, Topology: {}, Game: {}, Dynamics: {}".format(N,h,w,C,V,topology,game,dynamics))

		#self.game_function()=game_list(self.game)


	def generate_p(self):
		"""Generates a matrix p_ij, NxM of the probability of finding each individual I_i in place P_m. This matrix is
		defined based on the topology of the underlying network, its size M, subpopulation size Q, and the home fidelity h."""

		p=np.zeros((self.M,self.M))
		for m_i in range(self.M):
			d=sum(self.adjacency_matrix[m_i])
			p[m_i]=self.adjacency_matrix[m_i]/float(self.h+d)
			p[m_i][m_i]=float(self.h)/float(self.h+d)
		#print(p)
		self.p=p


	def get_probability_group(self,m,G):
		"""Returns the probability of group G meeting at place P_m. The full computation involves a cycle computing the probability
		of finding each individual either in or out of the group, thus having N interations."""

		if max(G)>self.Q:
			#print(str(G)+ "  "+str(self.Q)+"  "+str(max(G)))
			print("ERROR: Subgroups larger than subpopulations.")

		chi=1.
		#for i in G:
		#	chi *= self.p[i,m]
		#for i in list(set(range(self.N))-set(G)):
		#	chi *= (1-self.p[i,m])

		for m_i in range(self.M):
			#print(self.p[m_i,m]**G[m_i] * (1-self.p[m_i,m])**(self.Q-G[m_i]))
			chi *= self.p[m_i,m]**G[m_i] * (1-self.p[m_i,m])**(self.Q-G[m_i])

		return chi



	def get_reward_PD(self,type,c,d):
		if type=="C":
			return -self.C + self.V*(c+1)/(c+d+1)
		return self.V*c/(c+d+1)

	def get_reward_PDV(self,type,c,d):
		sum_omega = 0
		if type=="C": 
			for i in range(c+1):
				sum_omega += self.omega**i
			return -self.C + self.V/(c+d+1)*sum_omega
		for i in range(c):
			sum_omega += self.omega**i
		return self.V/(c+d+1)*sum_omega

	def get_reward_CPD(self,type,c,d):
		if c==0:
			if type=="C":
				return -self.C 
			return 0.
		if type=="C":
			return - self.C + self.V*c/(c+d)
		return self.V*c/(c+d)
	
	def get_reward_SH(self,type,c,d):
		if type=="C":
			if c+1>=self.L:
				return self.V*(c+1)/(c+d+1)-self.C
			return - self.C
		if c>=self.L:
			return self.V*c/(c+d+1)
		return 0.
		
	def get_reward_FSH(self,type,c,d):
		if type=="C":
			if c+1>=self.L:
				return self.V/(c+d+1)-self.C
			return -self.C
		if c>=self.L:
			return self.V/(c+d+1)
		return 0.
		
	def get_reward_VD(self,type,c,d):
		if type=="C":
			return self.V-self.C
		if c>0:
			return self.V
		return 0.
		
	def get_reward_TVD(self,type,c,d):
		if type=="C":
			if c+1>=self.L:
				return self.V-self.C
			return -self.C
		if c>=self.L:
			return self.V
		return 0.
		
	def get_reward_SD(self,type,c,d):
		if type=="C":
			return self.V-self.C/(c+1)
		if c>0:
			return self.V
		return 0.
		
	def get_reward_TSD(self,type,c,d):
		if type=="C":
			if c+1>=self.L:
				return self.V - self.C/(c+1)
			return -self.C/self.L
		if c>=self.L:
			return self.V
		return 0.
		
	def get_reward_HD(self,type,c,d):
		if type=="D":
				return (self.V-d*self.C)/(d+1)
		if d==0:
			return self.V/(c+1)
		return 0.


	def get_fitness(self,strat1,m1,S):
		#could have to replace k by (self.M-k)
		s = int(np.dot(np.array(S),np.array([(self.Q+1)**k for k in range(self.M)])))
		if strat1=="C":
			#print(str(m1)+" " + str(s))
			return self.fitness[m1][s]
		elif strat1=="D":
			return self.fitness[m1+self.M][s]
		print("ERROR: Something is wrong.")
		return None


	def generate_fitness(self):
		"""Returns the fitness weighted fitness of an "individual" in place P_m and state S, taking into consideration all the
		possible groups they can form. This function is an intermediary for the get_fitness() one. Individuals in a group are
		accounted as "1"s from the last to the first digit in g_iter."""
		
		fitness=np.zeros((self.M*2,(self.Q+1)**self.M))



		for s_iter in range((self.Q+1)**self.M):

			S=[(s_iter//((self.Q+1)**k))%(self.Q+1) for k in range(self.M)]

			#print(S)


			#Simplification added for complete networks
			if self.topology=="complete" and S!=sorted(S,reverse=True):
				S_sorted=sorted(S,reverse=True)
				#print("Sorted: "+str(S_sorted))
				s_iter_sorted= sum([S_sorted[k]*(self.Q+1)**k for k in range(self.M)])

				for m in range(self.M):
					m_iter_sorted = S_sorted.index(S[m])
					fitness[m][s_iter]= fitness[m_iter_sorted][s_iter_sorted]
					fitness[m+self.M][s_iter]= fitness[m_iter_sorted+self.M][s_iter_sorted]

				continue
			#End of simplification added for complete networks

			for m_iter in range(self.M):
				if S[m_iter]==0:
					fitness[m_iter][s_iter]=None

				if S[m_iter]==self.Q:
					fitness[m_iter+self.M][s_iter]=None

			for m_iter in range(2*self.M):
				
				m1=m_iter%self.M
				
				if m_iter<self.M:
					strat1="C"
				else:
					strat1="D"


				for m in range(self.M):

					#Guarantee that individual with strat1 with home m1 is always in the group, and take
					#this into consideration in the calculation of repetitions and in the number of C/D in the group
					for c_iter in range((self.Q+1)**self.M):
						C=[(c_iter//((self.Q+1)**k))%(self.Q+1) for k in range(self.M)]
						C_aux=list(C)
						if strat1=="C" and C[m1]==0:
							continue
						if strat1=="C":
							C_aux[m1]=0

						if any([C[m_j]>S[m_j] for m_j in range(self.M)]):
							continue

						for d_iter in range((self.Q+1)**self.M):
							D=[(d_iter//((self.Q+1)**k))%(self.Q+1) for k in range(self.M)]
							D_aux=list(D)

							#print("c_iter: "+str(c_iter)+", C: "+str(C)+"; d_iter: "+str(d_iter)+", D: "+str(D))

							if strat1=="D" and D[m1]==0:
								continue
							if strat1=="D":
								D_aux[m1]=0
							
							if any([ D[m_j] > self.Q-S[m_j] for m_j in range(self.M)]):
								continue

							coef_factor  = np.prod([float(comb(S[i],C_aux[i])*comb(self.Q-S[i],D_aux[i])) for i in range(self.M)])

							if strat1=="C":
								coef_factor *= float(comb(S[m1]-1,C[m1]-1))
							elif strat1=="D":
								coef_factor *= float(comb(self.Q-S[m1]-1,D[m1]-1)) 
							
							G=np.array(C)+np.array(D)
							c=sum(C)
							d=sum(D)
							if strat1=="C":
								c-=1
							else:
								d-=1
							
							fitness[m_iter][s_iter]+= coef_factor * self.get_probability_group(m,G) * self.get_reward(strat1,c,d)
					
		#print(fitness)		
		#print(1. - self.w + self.w*fitness)
		self.fitness = 1. - self.w + self.w*fitness
		#return self.fitness


	def get_replacement_weight(self,m1,m2,test_same):

		if test_same!=True:
			return self.replacement_weights[m1][m2]
		if m1==m2:
			return self.self_replacement_weights[m1]
		print("ERROR: Something is wrong.")
		return None
		


	def generate_replacement_weights(self):
		"""Provides the replacement weights w_{individual1,individual2}. Individuals in a group are accounted as
		"1"s from the last to the first digit in g_iter."""

		self_replacement_weights=np.zeros(self.M)
		replacement_weights=np.zeros((self.M,self.M))

		if self.Q==1:
			np.fill_diagonal(replacement_weights,None)


		for m_j in range(self.M):
			G = [0]*self.M
			G[m_j]=1
			for m in range(self.M):
				#replacement_weight+= self.get_probability_group(m,[individual1])
				self_replacement_weights[m_j]+= self.get_probability_group(m,G)



		for m in range(self.M):
			#print("\nm: ",m)

			for g_iter in range((self.Q+1)**self.M):
				G=[(g_iter//((self.Q+1)**k))%(self.Q+1) for k in range(self.M)]
				#print("Printing tested groups:  "+str(G))

				if sum(G)<2:
					continue

				probability_group=self.get_probability_group(m,G)/(sum(G)-1.)


				for m1 in range(self.M):

					if G[m1]==0:
						continue

					for m2 in range(m1+1):

						if G[m2]==0 or (m1==m2 and G[m2]<2):
							continue

						G_aux=list(G)
						G_aux[m1]=0
						G_aux[m2]=0
						#print(scipy.misc.comb(self.Q,G_aux[0]))
						#print("group: "+str(G))
						coef_factor = np.prod([float(comb(self.Q,G_aux[i])) for i in range(self.M)])
						#print("1st factors: "+str([float(scipy.misc.comb(self.Q,G_aux[i])) for i in range(self.M)]))

						if m1!=m2:
							coef_factor *= float(comb(self.Q-1,G[m1]-1)) * float(comb(self.Q-1,G[m2]-1))
							#print("2nd factors: "+str(float(scipy.misc.comb(self.Q-1,G[m1]-1)) * float(scipy.misc.comb(self.Q-1,G[m2]-1))))
						else:
							coef_factor *= float(comb(self.Q-2,G[m1]-2))

						#print(G)
						#print(self.get_probability_group(m,G)/(sum(G)-1))
						replacement_weights[m1][m2]+= coef_factor * probability_group

		for m1 in range(self.M):
			for m2 in range(m1+1,self.M):
				replacement_weights[m1][m2]+= replacement_weights[m2][m1]

		self.self_replacement_weights=self_replacement_weights
		self.replacement_weights=replacement_weights





	def get_birth_parameter_BDB(self,strat1,m1,S):
		"""Computes and provides the birth parameter b_{individual} from the BDB dynamics, based on its fitness.Simplifications
		are made based on the repetition of fitness values in population."""

		# The general involves a sum over all values of fitness.

		f_sum=0.
		for m_j in range(self.M):
			if S[m_j]>0:
				f_sum += S[m_j]*self.get_fitness("C",m_j,S)
			if S[m_j]<self.Q:
				f_sum += (self.Q-S[m_j])*self.get_fitness("D",m_j,S)


		return self.get_fitness(strat1,m1,S) / (f_sum)


	def get_death_parameter_BDB(self,m1,m2,test_same):
		"""Provides the death parameter d_{individual1,individual2} from the BDB dynamics, based on i1's replacement weights."""

		return self.get_replacement_weight(m1,m2,test_same) #/ (f_sum)

	def get_replacement_probability_BDB(self,strat1,m1,strat2,m2,S,test_same=None):
		"""Provides the probability of i1 replacing i2 when population is in state S under BDB dynamics."""

		if strat1==strat2 and m1==m2 and self.Q>1 and test_same==None:
			print("ERROR: Something is wrong: calculating replacement probability without clarifing whether individuals are the same.")

		if (strat1=="C" and S[m1]==0) or (strat2=="C" and S[m2]==0) or (strat1=="D" and S[m1]==self.Q) or (strat2=="D" and S[m2]==self.Q):
			print("ERROR: Something is wrong: calculating replacement probability between inexistent individuals in the given state S.")			


		return self.get_birth_parameter_BDB(strat1,m1,S) * self.get_death_parameter_BDB(m1,m2,test_same)


	def get_death_parameter_BDD(self,strat1,m1,strat2,m2,S,test_same):
		"""Provides the death parameter d_{individual1,individual2} from the BDD dynamics, based on i2's weighted inversed fitness."""		

		wif_sum=0
		for m_j in range(self.M):
			if m1!=m_j:
				if S[m_j]>0:
					wif_sum += S[m_j] * self.get_replacement_weight(m1,m_j,False) / self.get_fitness("C",m_j,S)
				if S[m_j]<self.Q:
					wif_sum += (self.Q - S[m_j]) * self.get_replacement_weight(m1,m_j,False) / self.get_fitness("D",m_j,S)
			elif m1==m_j and strat1=="C":
				wif_sum += self.get_replacement_weight(m1,m_j,True) / self.get_fitness("C",m_j,S)
				if S[m_j]>1:
					wif_sum +=  (S[m_j]-1) * self.get_replacement_weight(m1,m_j,False) / self.get_fitness("C",m_j,S)
				if S[m_j]<self.Q:
					wif_sum +=  (self.Q - S[m_j]) * self.get_replacement_weight(m1,m_j,False) / self.get_fitness("D",m_j,S)
			elif m1==m_j and strat1=="D":
				wif_sum += self.get_replacement_weight(m1,m_j,True) / self.get_fitness("D",m_j,S)
				if S[m_j]>0:
					wif_sum +=  S[m_j] * self.get_replacement_weight(m1,m_j,False) / self.get_fitness("C",m_j,S)
				if S[m_j]<self.Q-1:
					wif_sum +=  (self.Q - S[m_j]-1) * self.get_replacement_weight(m1,m_j,False) / self.get_fitness("D",m_j,S)

			#print(self.get_replacement_weight(individual1,i)/ self.get_fitness(i,S))

		return (self.get_replacement_weight(m1,m2,test_same)/  self.get_fitness(strat2,m2,S)   ) / (wif_sum)

	def get_replacement_probability_BDD(self,strat1,m1,strat2,m2,S,test_same=None):
		"""Provides the probability that one particular individual with home in place m1 and using strategy strat1
		will replace one particular individual with home in place m2 and using strategy strat2 when the population 
		is in state S under BDD dynamics."""

		if strat1==strat2 and m1==m2 and self.Q>1 and test_same==None:
			print("ERROR: Something is wrong: calculating replacement probability without clarifing whether individuals are the same.")

		if (strat1=="C" and S[m1]==0) or (strat2=="C" and S[m2]==0) or (strat1=="D" and S[m1]==self.Q) or (strat2=="D" and S[m2]==self.Q):
			print("ERROR: Something is wrong: calculating replacement probability between inexistent individuals in the given state S.")			

		return (1./float(self.N))* self.get_death_parameter_BDD(strat1,m1,strat2,m2,S,test_same)


	def get_death_parameter_DBB(self,strat1,m1,strat2,m2,S,test_same):
		"""Provides the death parameter d_{individual1,individual2} from the DBB dynamics, based on i2's weighted fitness."""		

		wf_sum=0
		for m_j in range(self.M):
			if m2!=m_j:
				if S[m_j]>0:
					wf_sum += S[m_j] * self.get_replacement_weight(m_j,m2,False) * self.get_fitness("C",m_j,S)
				if S[m_j]<self.Q:
					wf_sum += (self.Q - S[m_j]) * self.get_replacement_weight(m_j,m2,False) * self.get_fitness("D",m_j,S)
			elif m2==m_j and strat2=="C":
				wf_sum += self.get_replacement_weight(m_j,m2,True) * self.get_fitness("C",m_j,S)
				if S[m_j]>1:
					wf_sum +=  (S[m_j]-1) * self.get_replacement_weight(m_j,m2,False) * self.get_fitness("C",m_j,S)
				if S[m_j]<self.Q:
					wf_sum +=  (self.Q - S[m_j]) * self.get_replacement_weight(m_j,m2,False) * self.get_fitness("D",m_j,S)
			elif m2==m_j and strat2=="D":
				wf_sum += self.get_replacement_weight(m_j,m2,True) * self.get_fitness("D",m_j,S)
				if S[m_j]>0:
					wf_sum +=  S[m_j] * self.get_replacement_weight(m_j,m2,False) * self.get_fitness("C",m_j,S)
				if S[m_j]<self.Q-1:
					wf_sum +=  (self.Q - S[m_j]-1) * self.get_replacement_weight(m_j,m2,False) * self.get_fitness("D",m_j,S)

			#print(self.get_replacement_weight(individual1,i)/ self.get_fitness(i,S))

		return (self.get_replacement_weight(m1,m2,test_same) *  self.get_fitness(strat1,m1,S)   ) / (wf_sum)

	def get_replacement_probability_DBB(self,strat1,m1,strat2,m2,S,test_same=None):
		"""Provides the probability that one particular individual with home in place m1 and using strategy strat1
		will replace one particular individual with home in place m2 and using strategy strat2 when the population 
		is in state S under DBB dynamics."""

		if strat1==strat2 and m1==m2 and self.Q>1 and test_same==None:
			print("ERROR: Something is wrong: calculating replacement probability without clarifing whether individuals are the same.")

		if (strat1=="C" and S[m1]==0) or (strat2=="C" and S[m2]==0) or (strat1=="D" and S[m1]==self.Q) or (strat2=="D" and S[m2]==self.Q):
			print("ERROR: Something is wrong: calculating replacement probability between inexistent individuals in the given state S.")			

		return (1./float(self.N))* self.get_death_parameter_DBB(strat1,m1,strat2,m2,S,test_same)


	def get_transition_probability(self,S,S1):
		"""Provides the probability of the population transitioning from state S to state S1. This involves four branches.
		Complete graph does not need to use this function, because its excplicit form solution uses replacement probabilities
		directly."""

		#print("ERROR: ADD SECTIONS WITH THE POSSIBILITY THAT INDIVIDUALS ARE THE SAME.")

		#If state is the same, sums over the probability of i replacing j, given that they are the same strategy.
		if S==S1:
			p=0.

			for m_parent in range(self.M):
				n_C_parent=S[m_parent]
				if n_C_parent > 0:
					for m_dead in range(self.M):
						n_C_dead = S[m_dead]
						if n_C_dead > 0 and m_dead!=m_parent:
							p += n_C_parent  * n_C_dead * self.get_replacement_probability("C",m_parent,"C",m_dead,S)
						elif m_dead==m_parent:
							p += n_C_parent * self.get_replacement_probability("C",m_parent,"C",m_dead,S,test_same=True)
							if n_C_dead > 1:
								p += n_C_parent * (n_C_dead -1) * self.get_replacement_probability("C",m_parent,"C",m_dead,S,test_same=False)

				n_D_parent = self.Q - S[m_parent]
				if n_D_parent > 0:
					for m_dead in range(self.M):
						n_D_dead = self.Q - S[m_dead]
						if	n_D_dead > 0 and m_dead!=m_parent:
							p += n_D_parent  * n_D_dead * self.get_replacement_probability("D",m_parent,"D",m_dead,S)
						elif m_dead==m_parent:
							p += n_D_parent * self.get_replacement_probability("D",m_parent,"D",m_dead,S,test_same=True)
							if n_D_dead > 1:
								p += n_D_parent * (n_D_dead -1) * self.get_replacement_probability("D",m_parent,"D",m_dead,S,test_same=False)

			return p



		#If S1 has an extra individual when compared to S, it sums over the probability of i in S replacing j not in S.
		#Because S is smaller than S1, if some elements do not coincide, set(S1)-set(S) is larger than 1.
		if sum(S1)>sum(S) and sum(abs(np.array(S1)-np.array(S)))==1:
			m_dead = list(np.array(S1)-np.array(S)).index(1)
			p = 0.
			#print(m_born,np.array(S1)-np.array(S))
			#print(S,S1)
			n_dead_D = self.Q - S[m_dead]
			for m_parent in range(self.M):
				n_parent_C = S[m_parent]
				if n_parent_C  > 0:
					p += n_parent_C  * n_dead_D * self.get_replacement_probability("C",m_parent,"D",m_dead,S,test_same=False)
			return p

		#If S1 has one less individual when compared to S1, sums over the probability of i not in S replacing j in S.
		if sum(S)>sum(S1) and sum(abs(np.array(S)-np.array(S1)))==1:
			m_dead = list(np.array(S1)-np.array(S)).index(-1)
			p = 0.
			n_dead_C = S[m_dead]
			for m_parent in range(self.M):
				n_parent_D = self.Q - S[m_parent]
				if n_parent_D > 0:
					p += n_parent_D * n_dead_C * self.get_replacement_probability("D",m_parent,"C",m_dead,S,test_same=False)
			return p
		
		#Otherwise is zero
		return 0.

	def get_arithmetic_fixation_probabilities(self):
		"""Provides the aritmetic average of the probability of [one single agent A fixating on a population of B; and
		one single agent B fixating on a population of A]."""

		#time1=time.time()

		#Generally, we can use a solver of linear equations to get the fixation probability. This set has 2^N equations (32 for N=5,
		#1024 for N=10) with variable being the fixation probabilities of A in state S.
		a=np.zeros(((self.Q+1)**self.M,(self.Q+1)**self.M))
		b=np.zeros((self.Q+1)**self.M)

		b[(self.Q+1)**self.M-1]=1.

		np.fill_diagonal(a,1.)


		for s_iter in range(1,(self.Q+1)**self.M-1):
			S=[(s_iter//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)]
			#print(S)
			for s1_iter in range(0,(self.Q+1)**self.M):
				S1=[(s1_iter//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)]
				a[s_iter][s1_iter] -= self.get_transition_probability(S,S1)


		#time2=time.time()
		#print("Time to build transition matrix: "+str(time2-time1))

		#print(a)
		c = np.linalg.solve(a,b)
		#print(c)

		#time3=time.time()
		#print("Time to solve system of equations: "+str(time3-time2))

		f_prob_A=0.
		#We are averaging over the states with just one strategy A: 0001, 0010, 0100, 1000
		for i in range(self.M):
			f_prob_A += c[(self.Q+1)**i]
			#print([((self.Q+1)**i/((self.Q+1)**m))%(self.Q+1) for m in range(self.M)])
		f_prob_A /= float(self.M)

		f_prob_B=0.
		#We are averaging over the states with just one strategy B: 1110, 1101, 1011, 0111
		for i in range(self.M):
			f_prob_B += 1-c[(self.Q+1)**self.M-1-(self.Q+1)**i]
		f_prob_B /= float(self.M)



		return [f_prob_A, f_prob_B]


	def get_fixation_probabilities(self):
		"""Provides both the aritmetic and the temperature-weighted average of the probability of one single 
		mutant using A fixating on a population of B and vice-versa. Result in the form of
		[[fix_A, fix_B],[fixT_A, fixT_B]]"""


		#We solve the linear system represented by a*x=b for x the fixation probabilities starting from
		#each state of the population. After solving the system of equation, seek the particular entries
		#of x corresponding to starting from states with single individuals using A or B.

		a=np.zeros(((self.Q+1)**self.M,(self.Q+1)**self.M))
		b=np.zeros((self.Q+1)**self.M)

		b[(self.Q+1)**self.M-1]=1.

		np.fill_diagonal(a,1.)


		for s_iter in range(1,(self.Q+1)**self.M-1):
			S=[(s_iter//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)]
			count=0.
			for s1_iter in range(0,(self.Q+1)**self.M):
				S1=[(s1_iter//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)]
				a[s_iter][s1_iter] -= self.get_transition_probability(S,S1)
				count+=self.get_transition_probability(S,S1)

		c = np.linalg.solve(a,b)


		#print("c:\n"+str(c))
		#print("1-c:\n"+str(np.array([1.]*(self.Q+1)**self.M)-c))

		fixT_A=0.
		fix_A=0.
		sum_temp=0.
		#We are averaging over the states with just one strategy A: 1000, 0100, 0010, 0001
		for i in range(self.M):
			#t=self.get_strict_temperature(self.M-i-1)
			t=self.get_strict_temperature(i)
			fixT_A += t*c[(self.Q+1)**i]
			sum_temp += t
			fix_A += float(1./self.M)*c[(self.Q+1)**i]
		fixT_A /= sum_temp

		fixT_B=0.
		fix_B=0.
		#We are averaging over the states with just one strategy B: (Q-1).Q.Q.Q, Q.(Q-1).Q.Q, Q.Q.(Q-1).Q, Q.Q.Q.(Q-1)
		for i in range(self.M):
			#fixT_B += self.get_strict_temperature(self.M-i-1)*(1.-c[(self.Q+1)**self.M-1-(self.Q+1)**i])
			fixT_B += self.get_strict_temperature(i)*(1.-c[(self.Q+1)**self.M-1-(self.Q+1)**i])
			fix_B += float(1./self.M)*(1.-c[(self.Q+1)**self.M-1-(self.Q+1)**i])
		fixT_B /= sum_temp


		#print(float(fix_A))
		return [[float(fix_A), float(fix_B)],[float(fixT_A), float(fixT_B)]]

	def get_weighted_fixation_probabilities_B(self):
		"""Provides the aritmetic average of the probability of [one single agent A fixating on a population of B; and
		one single agent B fixating on a population of A]."""

		#time1=time.time()

		#Generally, we can use a solver of linear equations to get the fixation probability. This set has 2^N equations (32 for N=5,
		#1024 for N=10) with variable being the fixation probabilities of A in state S.
		a=np.zeros(((self.Q+1)**self.M,(self.Q+1)**self.M))
		b=np.zeros((self.Q+1)**self.M)

		b[0]=1.

		np.fill_diagonal(a,1.)


		for s_iter in range(1,(self.Q+1)**self.M-1):
			S=[(s_iter//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)]
			#print(S)
			for s1_iter in range(0,(self.Q+1)**self.M):
				S1=[(s1_iter//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)]
				a[s_iter][s1_iter] -= self.get_transition_probability(S,S1)

		print("a:\n"+str(a))

		#time2=time.time()
		#print("Time to build transition matrix: "+str(time2-time1))

		#print(a)
		c = np.linalg.solve(a,b)
		#print(c)

		print("c:\n"+str(c))

		print("1-c:\n"+str(np.array([1.]*(self.Q+1)**self.M)-c))

		#time3=time.time()
		#print("Time to solve system of equations: "+str(time3-time2))

		f_prob_A=0.
		sum_temp=0.
		#We are averaging over the states with just one strategy A: 1000, 0100, 0010, 0001
		for i in range(self.M):
			t=self.get_strict_temperature(i)
			#print(t)
			f_prob_A += t*(1-c[(self.Q+1)**i])
			#print([ ((self.Q+1)**i//((self.Q+1)**m))%(self.Q+1) for m in range(self.M)])
			sum_temp += t
			#print([((self.Q+1)**i/((self.Q+1)**m))%(self.Q+1) for m in range(self.M)])
		f_prob_A /= sum_temp
		#print(sum_temp)

		f_prob_B=0.
		#We are averaging over the states with just one strategy B: (Q-1).Q.Q.Q, Q.(Q-1).Q.Q, Q.Q.(Q-1).Q, Q.Q.Q.(Q-1)
		for i in range(self.M):
			f_prob_B += self.get_strict_temperature(i)*c[(self.Q+1)**self.M-1-(self.Q+1)**i]
			#print([( ((self.Q+1)**self.M-1-(self.Q+1)**i)//((self.Q+1)**m))%(self.Q+1)for m in range(self.M)])
		f_prob_B /= sum_temp



		return [f_prob_A, f_prob_B]



	def get_in_temperature(self, individual):
		"""Provides the in temperature of an "individual". Should be always 1."""
		temperature=0.
		for j in range(self.M):
			if individual==j:
				continue
			temperature+=self.Q*self.replacement_weights[j,individual]
		temperature += self.self_replacement_weights[individual]
		if self.Q>1:
			temperature += (self.Q-1) * self.replacement_weights[individual,individual]
		return temperature

	
	def get_out_temperature(self, individual):
		"""Provides the out temperature of an "individual". Should be always 1."""
		temperature=0.
		for j in range(self.M):
			if individual==j:
				continue
			temperature+=self.Q*self.replacement_weights[individual,j]
		temperature += self.self_replacement_weights[individual]
		if self.Q>1:
			temperature += (self.Q-1) * self.replacement_weights[individual,individual]
		return temperature

	def get_strict_temperature(self, individual):
		"""Provides the strict temperature of an "individual". Is a measure of how likely this individual is of replacing someone/
		being replaced."""
		#Under the complete graph, we can use the fact that every weight w_{ij}, with i!=j is the same and every weight w_{kk} are
		#also the same.
		#print(1. - self.self_replacement_weights[individual])
		return (1. - self.self_replacement_weights[individual])



	def get_average_strict_temperature(self):
		"""Provides the strict temperature of an "individual". Is a measure of how likely this individual is of replacing someone/
		being replaced."""
		#Under the complete graph, we can use the fact that every weight w_{ij}, with i!=j is the same and every weight w_{kk} are
		#also the same.

		av=0.
		for i in range(self.M):
			av+=(1./self.M)*self. get_strict_temperature(i)

		return av

	def get_strict_subpopulation_temperature(self, m_j):
		"""Provides the strict temperature of an "individual". Is a measure of how likely this individual is of replacing someone/
		being replaced."""
		#Under the complete graph, we can use the fact that every weight w_{ij}, with i!=j is the same and every weight w_{kk} are
		#also the same.
		#print(1. - self.self_replacement_weights[individual])
		
		sstemp=0.
		for m_i in range(self.M):
			if m_i == m_j:
				continue
			sstemp+=float(self.Q)*float(self.Q)*self.replacement_weights[m_i,m_j]

		return sstemp


	def get_strict_subpopulation_temperature_2(self, m_j):
		"""Provides the strict temperature of an "individual". Is a measure of how likely this individual is of replacing someone/
		being replaced."""
		#Under the complete graph, we can use the fact that every weight w_{ij}, with i!=j is the same and every weight w_{kk} are
		#also the same.
		#print(1. - self.self_replacement_weights[individual])
		
		extra=0.
		if self.Q>1:
			extra=(float(self.Q)-1.)*self.replacement_weights[m_j,m_j]
		return float(self.Q)*(1. - self.self_replacement_weights[m_j]-extra)

	def get_average_strict_subpopulation_temperature(self):
		"""Provides the strict temperature of an "individual". Is a measure of how likely this individual is of replacing someone/
		being replaced."""
		#Under the complete graph, we can use the fact that every weight w_{ij}, with i!=j is the same and every weight w_{kk} are
		#also the same.

		av=0.
		for i in range(self.M):
			av+=(1./self.M)*self. get_strict_subpopulation_temperature(i)
		return av


	def get_average_group_size(self):
		"""Provides the replacement weights w_{individual1,individual2}. Individuals in a group are accounted as
		"1"s from the last to the first digit in g_iter."""


		av_gs=0.
		av_gs_2=0.

		
		for m in range(self.M):
			#print("\nm: ",m)

			for g_iter in range((self.Q+1)**self.M):
				G=[(g_iter//((self.Q+1)**k))%(self.Q+1) for k in range(self.M)]
			
				probability_group=self.get_probability_group(m,G)
				coef_factor = np.prod([float(comb(self.Q,G[i])) for i in range(self.M)])

				av_gs+= coef_factor *sum(G)* probability_group
				av_gs_2+= coef_factor *sum(G)*sum(G)* probability_group

		return av_gs_2/av_gs






