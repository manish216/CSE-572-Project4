Project4:
The Learning Task [Reinforcement Learning]
==========================
Manish Reddy Challamala,
December 6, 2018 ,manishre@buffalo.edu

For detailed explaination, please visit the below link:
[link for report pdf](https://github.com/manish216/CSE-572-Project4/blob/master/proj4.pdf)


## Abstract
To train the model to find the shortest path between tom and jerry by using reinforcement learning and Deep Q-Network.

## 1 Introduction

The goal of this project is to implement Reinforcement learning and Deep Q-
network to find the shortest path between the tom and jerry.
The code is implemented in by using two learning algorithms:

1. Neural Network [DNN].
2. Reinforcement Learning [RL].

## 2 Theory

## 2.1 Reinforcement Learning:

Reinforcement learning is a methods, where an agent learn how to behave in
a environment by performing actions and update the rewards for the actions
made in the environment.Here Environment is the object and agent is an Re-
inforcement learning algorithm[RL]. The simple form of reinforcement learning
algorithm is given below:

1. Firstly the environment sends the state to the agent and the agent takes
    the action to that state. The environment again sends the new state and
    reward for the previous task to the agent where the agent will update its
    knowledge of reward.
    Here
    Action (A) : possible moves that a agent can take.
    State(S) : Current Situation returned by environment.


```
Reward(R) : An immediate return send back from the environment toevaluate the last action.
Policy (π) : The strategy that the agent employs to determine ext action based on the current state.
value (V) : expected long term return with discount, as opposed to short term reward (R).
Reinforcement learning need to choose between exploration and exploitation.
Exploring is a process that takes random actions to calculate or obtain more training data.
Exploitation is a process that takes actions from current best version of learned policy.

```
## 2.2 Neural Network

The neural network model is a classification model which tries to predict output
y for a given input x

1. The neural network model contain two phase:
    1 Learning Phase
    2. Prediction Phase
2. In learning phase, the neural network takes inputs and corresponding outputs.
3. The neural network process the data and calculates the optimal weights to gain the training experience.
4. In prediction process the neural network is presented with the unseen data
    where it predicts a output by using its past training experience on that given data.
5. A neural network model can be build by using different layers:
    1. Dense Layer : A linear Operation in which every input is connected to every output by weights
    2. Convolution Layer: A linear operation in which input are connected to output by using subsets of weights of dense layer.

## 3 Coding Task:

### Neural Network:

1. The Deep Q-Network acts as a brain for our agent.
2. In this project I have implemented both Dense neural network.
3. For Dense neural network ,Creating a model with 3 layers, 
    1. input layer
    2. 2-hidden layer
    3. output layer
4. No of nodes for each layer is given below:
    1.No of nodes in input layer = Co-ordinates of Tom and Jerry
    2. No of nodes in hidden layer 128
    3.No of nodes in output layer 4
5. Activation functions used in hidden layer is relu [rectified linear unit] because
    it introduces the non linearity in the network and linear function is used on the output layer to predict the target class.
6. The input takes 4 values [position of player and fruit] which are then
    combined with respective to different actions [weights] to produce the 4 output values.
    The agent always choose such action which gives highest Q-value.
7. In neural network, by tuning the hyper-parameters like no of hidden nodes
    a, activation functions,optimizer and learning rate we can predict the bet-
    ter action values while exploring the environment which increases the ac-
    curacy.

```
Exponential Decay formula for epsilon:

self.epsilon = self.min_epsilon +((self.max_epsilon - self.min_epsilon) * math.exp(-(self.lamb*abs(self.steps))))
```
1. Epsilon value will determine at what probability a agent will take the
    random action. This epsilon value will take care off our exploitation and
    exploration trade-off.
2. we can tune the maximum and minimum epsilon values to get better trade-
    off between exploration and exploitation and can also change the function
    of exponential-epsilon decay function to linear or quadratic decay function.



```
Implement Q-function:

if(st_next is None):
t[act] = rew
else:
t[act] = rew +self.gamma*(np.argmax(q_vals_next[i]))
```
1. The above snippet is known as iterative Q-function.
2. In this snippet we are checking weather the next state reaches the goal. If
    it reaches we update the action with maximum reward value else we are
    updating the Q-values by discounting the reward with gamma.


## 3.1 Writing Task 1:

If agent always chooses the action that maximizes the Q-value, it means that we
are concentrating more on the Exploitation rather than exploring which leads
the agent to follow the same path even if have the some better optimal way
to follow. So to overcome this problem we are going for the exploration and
exploitation trade off.

### greedy

The agent believes to select the optimal path all the time but occasionally acts
randomly.The greedy determines the probability of taking random action, Due
to its random action the model concentrates on both exploitation and exploration.

### Bayesian Neural Network

Unlike the neural network, The Bayesian neural network acts probabilistically.Where
instead of having the single set of fixed weights. The BNN takes the probability
distribution over all weights.So by taking the probability distributions over the
weights will allow us to obtain the distribution over the actions in reinforcement
learning.The variance of of distribution will tell us about the uncertainty of each
action.

## 4 Writing Task 2:

Given:

Condition for sequence of actions: Right−→Down−→Right−→Down

| s0  | s1  | s5  |
| --- | --- | --- |
| s6  | s2  | s3  |
| s7  | s8  | s4  |

As we are calculating from the final state S4 will be updating the Q-table values
with Zero.

```

Q(S 4 ,U) = 0
Q(S 4 ,D) = 0
Q(S 4 ,L) = 0
Q(S 4 ,R) = 0
```

## ACTIONS

For Action caluclations please visit the report link given in section 6.


## 6. Graphs:

  The graph results are available here:
  [Results](https://github.com/manish216/CSE-572-Project4/blob/master/proj4.pdf)


