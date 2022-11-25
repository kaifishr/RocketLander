# RocketBooster ✨🚀✨

TL;DR: *RocketBooster* is a simple training environment to land an orbital rocket booster equipped using optimization algorithms such as Reinforcement Learning, Genetic Optimization, Evolution Strategies, and Simulated Annealing.

## Introduction

<p align="center">
    <img src="docs/booster.png" width="240" height=""/>
</p>

- The booster's goal is to reach the landing pad at a velocity smaller or equal to $v_{\text{max}}$.

Inspired by SpaceX's incredible progress, I set up a simple environment that uses PyBox2D for rigid physics simulation and PyGame for rendering and visualization, that allows to use different methods to land a booster in a physical simulation.

The physics is modeled using PyBox2D, a 2D physics library for rigid physics simulations. The booster consists of three sections. A long and low density section (the booster's hull containing mostly empty fuel tanks) connected to a short high density section (the booster's engines). On top of that there are the landing legs which are modeled as medium density sticks attached to the lower part of the rocket in a 45 degree angle.

The propulsive landing of the booster is learned using a neural network (the booster's brain) and one of the above mentioned optimization methods. The neural network controls the actions of the booster. The network's input (or state) at each time step consists of the booster's position ($r_x$, $r_y$), velocity ($v_x$, $v_y$), angle ($\theta$), and angular velocity ($\omega$). Based on the current state, the network predicts an action comprising thrust levels and engine angles.

## Run Examples

```console
cd rocketbooster
python -m projects.evolution_strategies.main
python -m projects.genetic_optimization.main
python -m projects.reinforcement_learning.main
python -m projects.simulated_annealing.main
```

## Methods


### Notation

In this project, the words *booster*, *agent*, *individual*, *candidate* are used interchangeably. 

So are *epoch* and *episode*.


### Reward Function

We use the same reward function for all optimization methods to measure the success of the agent during each episode. The reward function receives the current state of the environment as input and outputs a scalar value.

Building a good reward function leading to the expected behavior of an agent can be very challenging. Slight changes of the reward function can sometimes result in totally unexpected behavior of the agent.

A reward function for landing a rocket booster can be designed as follows. A first term captures the distance from the booster to the landing pad. The closer the booster is to the landing pad, the higher is the reward. This can be formulated as follows:

$$r_{\text{proximity}} = \frac{1}{1 + \alpha \sqrt{(x_\text{p} - x_\text{b})^2 + (y_\text{p} - y_\text{b})^2}}$$

with the $x$- and $y$-coordinates of the landing $\text p$ ad and the $\text b$ ooster. To avoid a rapid unscheduled disassembly of the booster, there is also a term that takes the booster's velocity into account,

$$r_{\text{velocity}} = \frac{1}{1 + \beta \sqrt{v_\text{x}^2 - v_\text{y}^2}}$$

with $x$- and $y$-components of the booster's velocity. $\alpha$ and $\beta$ are hyperparameter that allow to emphasis the reward coming from the proximity or velocity. By multiplying both terms we can encourage a soft landing and the final reward function is obtained:

$$R = r_{\text{proximity}} \cdot r_{\text{velocity}}$$

Using the formulation above of the reward function, the reward is always between 0 and 1.

Lowering the number of simulations steps equals a time restriction and at the same time resembles an implicit fuel restriction, encouraging the booster to land more quickly.


### Genetic Optimization

Inspired by evolution, genetic optimization (GO) uses a population of individuals that are slightly different from each other. These differences result from mutation which is a fundamental property of evolution and result in different behavior of each agent. The difference in behavior makes some agents more successful than others. The fitness or success of an agent is represented by the fitness or reward function.

The algorithm starts with a population of candidates, from which the agent with the highest fitness makes it to the next round. After the selection the fittest individual propagates its genes accompanied with random mutations on to the next generation. This process is repeated until the desired fitness is achieved.

Here, we use the mutation operation during the optimization process to learn to land a rocket booster. Mutation operations act on all members of a population. The mutation operation is defined by the probability with which a parameter mutation is going to happen and the rate or strength with which the mutation acts on the parameter.


### Evolution Strategies

Evolution Strategies (ES) is a class of black-box stochastic optimization techniques and has achieved some impressive results on RL benchmarks. Even though their names may suggest otherwise, evolution strategies optimization has very little similarity to genetic optimization. At its core, the ES optimization algorithm resembles simple hill-climbing in a high-dimensional space sampling a population of candidate solutions and allow agents with high reward to have a higher influence on the distribution of future generations.

Despite the simplicity, the ES algorithm is pretty powerful and overcomes many of RL's inconveniences. Optimization with ES is highly parallelizable, makes no assumptions about the underlying model to train, allows interactions between agents out of the box, and is not constraint to a discrete action space.

For the ES algorithm to work we only have to look at the parameterized reward function $R(\bm s; \bm \theta)$, takes takes a state vector and outputs a scalar reward. During the optimization process, we estimate gradients that allow us to steer the parameters $\bm \theta$ into a direction to maximize the reward $R$. Thus, we are optimizing $R$ with respect to $\bm \theta$.

The ES algorithm generates at each time step a population of different parameter configurations $\bm \theta_i$ (the agents / boosters neural network weights) from the base parameters $\bm \theta$ by adding gaussian noise ($\bm \theta_i = \bm \theta + \bm \epsilon$ with $\bm \epsilon \sim \mathcal{N}(0, \sigma^2)$ and $i \in [1, N]$). After each agent has spend one episode / epoch in the environment a weighted sum over each agents policy network's parameters and gained reward is being created. This weighted sum of parameter vectors becomes the new base parameters. Mathematically, ES uses finite differences along a few random directions at each optimization step to estimate the gradients of the reward function $R$ with respect to the parameter vector $\bm \theta$.


### Simulated Annealing

In 1983, Kirkpatrick et al., combined the insights of heating and cooling materials to change their physical properties with the Monte Carlo-based Metropolis-Hastings algorithm, to find approximate solutions to the traveling salesman problem. This combination led to the technique of Simulated Annealing (SA). It has been shown, that with a sufficiently high initial temperature and sufficiently long cooling time, the system's minimum-energy state is reached.

In a nutshell, simulated annealing selects at each iteration a randomly created candidate solution that is close to the current one under some distance metric. The system moves to the proposed solution either if it comes with a higher reward or with a temperature-dependent probability. With decreasing temperature, the temperature-dependent probability to accept worse solutions narrows and the optimization focuses more and more on improving solutions approaching a Monte Carlo algorithm behavior.

In a past project ([*NeuralAnnealing*](https://github.com/kaifishr/NeuralAnnealing)) I used SA for to optimize neural networks for a classification task. Therefore, I thought to give it a shot and use SA to optimize a network to learn how to land a booster.


### Reinforcement Learning

Reinforcement Learning (RL) is without a doubt one of the most interesting subfields of machine learning.

Reinforcement learning consists of an agent (here the booster) interacting with an environment whose actions follow a policy (the booster's neural network) that the agent learns over time.

At each time step, the agent observes the state of its environment and takes actions based on the policy the agent learned so far.

The result of each action carried out by the agent is associated with a reward and a transition to a new state.

The goal of RL is to learn a policy that allows to pick the best known actions at any state to maximize the reward received. 

Here we use Deep Q-Learning which is one of the core concepts in Reinforcement Learning (RL).

The implemented Deep Q-Learning algorithm uses a batch of episodes to learn a policy that maximizes the reward.

- NOTE: Run N agents in parallel and record their episodes.

- In Deep Q-Learning, a NN maps input states to pairs of actions and Q-values.

- We use a policy function (e.g. a neural network resembling the agent's brain), to compute what an agent is supposed to do in any given situation.

- The neural network takes the current state of its environment (position, velocity, angle, angular velocity) as input and outputs the probability of taking one of the allowed actions. 

- We can use Deep Q-Learning to learn a control policy to land our booster.

- Using Deep Q-Learning, we use a deep neural network to predict the expected utility (also Q-value) of executing an action in a given state.

- Training process
    - We start the training process with a random initialization of the policy (the neural network)
    - While the agent interacts with the environment, we record the produced data at each time step. These data are the current state, the agent's performed action, and the reward received.
    - Given the set of state-action-reward pairs, we can use backpropagation to encourage state-actions pairs that resulted in a positive or high reward discourage pairs with negative or low reward.
    - During the training process, we enforce a certain degree of exploration by injecting noise to the actions of the agent. Exploration is induced by sampling from the action distribution at each time step. This is in contrast to ES, where noise is not injected into the agent's action space, but rather directly in the parameter space.


#### Deep Q-Learning

- Deep Q-Learning, Policy Gradients are model-free learning algorithms as they do not use the transition probability distribution (and the reward function) associated with the Markov decision process (MDP), which, in RL, represents the problem to be solved. That means, RL algorithms do not learn a model of their environment's transition function to make predictions of future states and rewards.

- Model-free RL always needs to take an action before it can make predictions about the next state and reward.

- Model-free RL means, that the agent does not have access to a model of the environment. Here, the environment is a function used to predict state transition and rewards.

- Deep Q-Learning uses a trial and error method to learn about the environment it interacts with. This is also called exploration. 

- Q-Value is the maximum expected reward an agent can reach by taken a certain action $A$ in state $S$.

#### Memory Size

- Replay

#### Action Space

We select an action from a discrete action space (maximum thrust of an engine at a certain angle). At maximum thrust (only on or off), the discrete action space of the booster for five different angles covering $\{-10°,-5°,0°,5°,10°,\}$ looks as follows:

|#|Action|
|:---:|:---:|
|1|Main engine fire at -10°|
|2|Main engine fire at -5°|
|3|Main engine fire at 0°|
|...|...|
|6|Left engine fire at -10°|
|...|...|
|15|Right engine fire at 10°|


As we select one action a time, the action is an array with shape `(1,)`. For the action space above, the action can take values in the range from 0 to 14.

During training, we either choose a random action from our discrete action space, or we take the action with highest predicted utility predicted by the neural network for a given state.

#### State Space

The state (or observation) space is defined by the booster's position, velocity, angle, and angular velocity:

|Number|Observation|Minimum|Maximum|
|:---:|:---:|:---:|:---:|
|1|Position $r_x$|$r_{x,{\text{min}}}$|$r_{x,{\text{max}}}$|
|2|Position $r_y$|$r_{y,{\text{min}}}$|$r_{y,{\text{max}}}$|
|3|Velocity $v_x$|$v_{x,{\text{min}}}$|$v_{x,{\text{max}}}$|
|4|Velocity $v_y$|$v_{y,{\text{min}}}$|$v_{y,{\text{max}}}$|
|5|Angle $\theta$|$\theta_{\text{min}}$|$\theta_{\text{max}}$|
|6|Angular velocity $\omega$|$\omega_{\text{min}}$|$\omega_{\text{max}}$|

The ranges above are defined in the *config.yml* file.

Thus, an observation is an array with shape `(6,)`.

captures the booster's states within the simulation and is defined by the 

#### Parameters

- `epsilon` is the epsilon-greedy value. This value defines the probability, that the agent selects a random action instead of the action that maximizes the expected utility (Q-value).

- `decay_rate` determines the decay of the `epsilon` value after each epoch.

- `epsilon_min` minimum allowed value for `epsilon` during the training period.

- `gamma` is a discount factor that determines how much the agent considers future rewards.



## Installation

To run *RocketBooster*, install the latest master directly from GitHub. For a basic install, run:

```bash
git clone https://github.com/kaifishr/RocketBooster
cd RocketBooster 
pip install -r requirements.txt
```

Then start the optimization by running:
Start the training with genetic optimization by running the following command:

```bash
python -m projects.genetic_optimization.main
```

## TODOs

- Add project with deep reinforcement learning.
- Add fuel constraint.
- Remove engines running detection with new final reward.


## References

[1] [PyBox2D](https://github.com/pybox2d/pybox2d) on GitHub

[2] [backends](https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples/backends) for PyBox2D

[3] PyBox2D [tutorial](https://github.com/pybox2d/cython-box2d/blob/master/docs/source/getting_started.md)

[4] [Minimal PyBox2D examples](https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples)

[5] Box2D C++ [documentation](https://box2d.org/documentation/)

[6] OpenAI Blog, [*Evolution Strategies as a Scalable Alternative to Reinforcement Learning*](https://openai.com/blog/evolution-strategies/)

[7] Salimans et al., 2017, [*Evolution Strategies as a Scalable Alternative to Reinforcement Learning*](https://arxiv.org/abs/1703.03864)


## Citation

If you find this project useful, please use BibTeX to cite it as:

```bibtex
@misc{fischer2022rocketbooster,
  title={RocketBooster},
  author={Fischer, Kai},
  year={2022},
  howpublished={\url{https://github.com/kaifishr/RocketBooster}}
}
```

```bibtex
@article{fischer2022rocketbooster,
  title   = "RocketBooster",
  author  = "Fischer, Kai",
  journal = "GitHub repository",
  year    = "2022",
  month   = "Nov",
  url     = "https://github.com/kaifishr/RocketBooster"
}
```

## License

MIT