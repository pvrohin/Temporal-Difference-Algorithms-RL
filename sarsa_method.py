import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from random import random, choice


World = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0,0,0],
				  [0,1,1,1,1,1,1,1,1,1,1,0]])

rewards = np.zeros((4,12))

states = np.zeros((4,12,2),dtype=np.int16)

for i in range(World.shape[0]):
	for j in range(World.shape[1]):

		states[i][j][0] = i
		states[i][j][1] = j
		if World[i][j]==1:
			rewards[i][j] = -100
		else:
			rewards[i][j] = -1





states = np.reshape(states,(4*12,2)) # flattening for ease of parsing



Q = {}
#initializing Q(s,a). At edges actions possible are appropriately trimmed
for state in states:
	actions = [[1,0],[-1,0],[0,1],[0,-1]]

	if(state[0] == 0):
		if(state[1]>0 and state[1]<11):
			actions = [[1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[1,0],[0,1]]
		elif(state[1]==11):
			actions = [[1,0],[0,-1]]
	elif(state[0]==3):
		if(state[1]>0 and state[1]<11):
			actions = [[-1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[-1,0],[0,1]]
		elif(state[1]==11):
			actions = [[-1,0],[0,-1]]

	elif(state[1]==0):
		actions = [[-1,0],[1,0],[0,1]]

	elif(state[1]==11):
		actions = [[-1,0],[1,0],[0,-1]]

	for a in actions:
		
		Q['['+str(state[0])+ ' ' + str(state[1])+']' +','+str(a)] = 0


actions = [[1,0],[-1,0],[0,1],[0,-1]]

#Epsilon-greedy policy
def action_policy(state,Q,epsilon):
	max_val = -float('inf')
	actions = [[1,0],[-1,0],[0,1],[0,-1]]

	#If on edge, restrict the choice of actions feasible (A(s))
	if(state[0] == 0):
		if(state[1]>0 and state[1]<11):
			actions = [[1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[1,0],[0,1]]
		elif(state[1]==11):
			actions = [[1,0],[0,-1]]
	elif(state[0]==3):
		if(state[1]>0 and state[1]<11):
			actions = [[-1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[-1,0],[0,1]]
		elif(state[1]==11):
			actions = [[-1,0],[0,-1]]

	elif(state[1]==0):
		actions = [[-1,0],[1,0],[0,1]]

	elif(state[1]==11):
		actions = [[-1,0],[1,0],[0,-1]]


	action_to_take = [0,0]


	if random() > epsilon:
		for a in actions:
			if(Q['['+str(state[0])+ ' ' + str(state[1])+']' + ','+str(a)] > max_val):
				action_to_take = a
				max_val = Q['['+str(state[0])+ ' ' + str(state[1])+']' + ','+str(a)]

	else:
		action_to_take = choice(actions)

	return action_to_take


#Going to new state given current state and action
def new_state(state,action):
	s_new = [0,0]
	s_new[0] = state[0] + action[0]
	s_new[1] = state[1] + action[1]
	return s_new


#Hyperparameters
num_loops = 5000
alpha = 0.05
e = 0.2
gamma = 0.90
decay_rate = 1.25 # if rate is zero, then no decay

reward_per_episode = []
reward_sum = 0
hund_episodes = []


# Start training
for episode in range(num_loops):
	init_state = [3,0]

	epsilon = e/np.power((episode+1),decay_rate) #update epsilon based on decay rate

	s = init_state
	termination = False
	action_taken = action_policy(init_state,Q,epsilon) #Choose action based on epsilon greed policy (S,A)


	while not termination:
		s_new = new_state(s,action_taken) #Based on the action go to the new state

		reward = rewards[s_new[0]][s_new[1]] #Collect reward at that state (S,A,R)
		reward_sum += reward

		#If new state is cliff then take the appropriate reward and go to init state, if it is the end then set termination to true
		if s_new[0] ==3:
			if s_new[1] == 11:
				termination = True
			elif s_new[1] == 0:
				pass
			else:
				s_new[1] = 0

		#Having reached the new state we are at: (S,A,R,S)


		action_taken_from_new_state = action_policy(s_new,Q,epsilon) #Choose action based on epsilon greed policy (S,A,R,S,A)

		#Update Q based on, ON policy
		Q['['+str(s[0])+ ' ' + str(s[1])+']' + ','+str(action_taken)] = Q['['+str(s[0])+ ' ' + str(s[1])+']' + ','+str(action_taken)] + alpha*(reward + gamma*Q['['+str(s_new[0])+ ' ' + str(s_new[1])+']' + ','+str(action_taken_from_new_state)] - Q['['+str(s[0])+ ' ' + str(s[1])+']' + ','+str(action_taken)])

		#S<- S', a <- a'
		s = s_new
		action_taken = action_taken_from_new_state


	if((episode+1)%100 == 0):
		reward_per_episode.append(reward_sum/100)
		reward_sum = 0
		hund_episodes.append(episode)







### Plotting and printing ###

for state in states:
	actions = [[1,0],[-1,0],[0,1],[0,-1]]

	if(state[0] == 0):
		if(state[1]>0 and state[1]<11):
			actions = [[1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[1,0],[0,1]]
		elif(state[1]==11):
			actions = [[1,0],[0,-1]]
	elif(state[0]==3):
		if(state[1]>0 and state[1]<11):
			actions = [[-1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[-1,0],[0,1]]
		elif(state[1]==11):
			actions = [[-1,0],[0,-1]]

	elif(state[1]==0):
		actions = [[-1,0],[1,0],[0,1]]

	elif(state[1]==11):
		actions = [[-1,0],[1,0],[0,-1]]

	for a in actions:

		print(Q['['+str(state[0])+ ' ' + str(state[1])+']' +','+str(a)],' |', end=" ")

	print('\n')




def max_policy(state,Q):
	max_val = -float('inf')
	actions = [[1,0],[-1,0],[0,1],[0,-1]]

	if(state[0] == 0):
		if(state[1]>0 and state[1]<11):
			actions = [[1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[1,0],[0,1]]
		elif(state[1]==11):
			actions = [[1,0],[0,-1]]
	elif(state[0]==3):
		if(state[1]>0 and state[1]<11):
			actions = [[-1,0],[0,1],[0,-1]]
		elif(state[1]==0):
			actions = [[-1,0],[0,1]]
		elif(state[1]==11):
			actions = [[-1,0],[0,-1]]

	elif(state[1]==0):
		actions = [[-1,0],[1,0],[0,1]]

	elif(state[1]==11):
		actions = [[-1,0],[1,0],[0,-1]]


	action_to_take = [0,0]



	for a in actions:
		if(Q['['+str(state[0])+ ' ' + str(state[1])+']' + ','+str(a)] > max_val):
			action_to_take = a
			max_val = Q['['+str(state[0])+ ' ' + str(state[1])+']' + ','+str(a)]


	return action_to_take



termination = False

state = [3,0]

while not termination:
	direction = max_policy(state,Q)
	plt.arrow(state[1],state[0],direction[1],direction[0],head_width=0.1)

	state = new_state(state,direction)

	if state[0] ==3 and state[1] == 11:
		termination = True


[x,y] = np.shape(World)

output = list(np.zeros(np.shape(World)))

for i in range(x):
	for j in range(y):
		if (World[i][j] == 0):
			output[i][j] = 1


plt.imshow(output,cmap='gray')
plt.show() 


plt.plot(hund_episodes,reward_per_episode)
plt.show()