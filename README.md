# **FalconLander** ✨🚀✨
# **RocketLander** ✨🚀✨
# **BoosterLander** ✨🚀✨

---

# Introduction

*RocketLander* is a comprehensive framework equipped with optimization algorithms, such as reinforcement learning, evolution strategies, genetic optimization, and simulated annealing, to enable an orbital rocket booster to land autonomously. *RocketLander* is designed to be simple to use and can be easily extended. 

<p align="center">
    <img src="docs/booster.png" width="240" height=""/>
</p>

$\textbf{\text{\textcolor{red}{TODO: Replace image with short gif of a landing booster.}}}$

The framework uses [*PyBox2D*](https://box2d.org/) a 2D physics library for rigid physics simulations, and [*PyGame*](https://www.pygame.org/) for rendering and visualization. 

I tried to make the simulation relatively realistic, even though that may conflict with [Box2D's recommendation](https://box2d.org/documentation/index.html#autotoc_md17) on object sizes. The booster has a height of about $46$ meters, a weight of about $25$ metric tons, and is made up of three parts. A long and low-density hull section containing mostly empty fuel tanks, a short but high-density engine section, and static medium-density landing legs.

In this framework, a booster is considered an agent that is equipped with a neural network (the agent's "brain") to learn how to propulsively land itself. The network is trained using reinforcement learning, evolution strategies, genetic optimization, or simulated annealing.

The neural network controls the actions of the booster. At each time step, the network receives the current state of the booster, which includes its position ($r_x$, $r_y$), velocity ($v_x$, $v_y$), angle ($\theta$), and angular velocity ($\omega$), as input. Based on this information, the network predicts an action, such as the levels of thrust and engine deflection.


# Installation

To run *RocketLander*, install the latest master directly from GitHub. For a basic install, run:

```bash
git clone https://github.com/kaifishr/RocketLander
cd RocketLander 
pip install -r requirements.txt
```

To start a training session using a specified learning method, run one of the examples in the project folder. For example:

```console
cd rocketlander
python -m projects.reinforcement_learning.main
python -m projects.evolution_strategies.main
python -m projects.genetic_optimization.main
python -m projects.simulated_annealing.main
```


# Methods

## Notation

In this project, the terms *booster*, *agent*, *individual*, and *candidate* are used interchangeably. Similarly, the terms *epoch* and *episode* are also used interchangeably. In the context of genetic optimization the terms *fitness* and *reward* are considered the same thing.


## Reward Function

Independent of the optimization method the same reward function is used to measure the success of the agent during each episode. The reward function receives the booster's current position and velocity as input and outputs a scalar value. A simple reward function for landing a rocket booster can be designed as follows. 

To encourage the booster to land as close as possible to the center of the landing pad, we can assign a high reward for proximity. For example, we can assign a reward of $1$ for a landing at the center of the landing pad, and reduce the reward to $0$ as the distance between the booster and the landing pad increases. This can be formulated as follows:

$$R_{\text{proximity}} = \frac{1}{1 + \alpha \sqrt{(r_{x, \text{pad}} - r_{x, \text{booster}})^2 + (r_{y, \text{pad}} - r_{y, \text{booster}})^2}}$$

with the $x$- and $y$-coordinates of the landing pad and the booster. To avoid a rapid unscheduled disassembly of the booster, there is also a term that takes the booster's velocity into account,

$$R_{\text{velocity}} = \frac{1}{1 + \beta \sqrt{v_\text{x}^2 - v_\text{y}^2}}$$

with the $x$- and $y$-components of the booster's velocity. The reward goes to $0$ for high velocities and to $1$ if the booster is not moving. The hyperparameters, $\alpha$ and $\beta$, allow us to emphasize the rewards coming from proximity or velocity. By multiplying these terms together, we obtain a reward function that ranges from $0$ to $1$ and encourages a soft landing at the center of the landing pad:

$$R = R_{\text{proximity}} \cdot R_{\text{velocity}}$$

We can implicitly model a fuel restriction by lowering the number of simulation steps. This time restriction resembles an implicit fuel restriction, encouraging the booster to land more quickly.


## Reinforcement Learning

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


### Deep Q-Learning

- Deep Q-Learning, Policy Gradients are model-free learning algorithms as they do not use the transition probability distribution (and the reward function) associated with the Markov decision process (MDP), which, in RL, represents the problem to be solved. That means, RL algorithms do not learn a model of their environment's transition function to make predictions of future states and rewards.

- Model-free RL always needs to take an action before it can make predictions about the next state and reward.

- Model-free RL means, that the agent does not have access to a model of the environment. Here, the environment is a function used to predict state transition and rewards.

- Deep Q-Learning uses a trial and error method to learn about the environment it interacts with. This is also called exploration. 

- Q-Value is the maximum expected reward an agent can reach by taken a certain action $A$ in state $S$.

### Memory Size

- Replay

### Action Space

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

### State Space

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


### Parameters

- `epsilon` is the epsilon-greedy value. This value defines the probability, that the agent selects a random action instead of the action that maximizes the expected utility (Q-value).

- `decay_rate` determines the decay of the `epsilon` value after each epoch.

- `epsilon_min` minimum allowed value for `epsilon` during the training period.

- `gamma` is a discount factor that determines how much the agent considers future rewards.


## Evolution Strategies

Evolution strategies is a class of black-box stochastic optimization techniques that have achieved impressive results on reinforcement learning benchmarks. Despite their name, evolution strategies optimization has very little in common with genetic optimization. At its core, the evolution strategies optimization algorithm resembles simple hill-climbing in a high-dimensional space. It samples a population of candidate solutions and allows agents with high rewards to have a greater influence on the distribution of future generations.

Despite the simplicity, the evolution strategies algorithm is pretty powerful and overcomes many of reinforcement learning inconveniences. Optimization with evolution strategies is highly parallelizable, makes no assumptions about the underlying model to train, allows interactions between agents by default, and is not restricted to a discrete action space.

For the evolution strategies algorithm to work we only have to look at the parameterized reward function $R(\bm s; \bm \theta)$, that takes a state vector and outputs a scalar reward. During the optimization process, we estimate gradients that allow us to steer the parameters $\bm \theta$ into a direction to maximize the reward $R$. Thus, we are optimizing $R$ with respect to $\bm \theta$.

The evolution strategies algorithm generates at each time step a population of different parameter configurations $\bm \theta_i$ (the agents' neural network weights) from the base parameters $\bm \theta$ by adding gaussian noise ($\bm \theta_i = \bm \theta + \bm \epsilon$ with $\bm \epsilon \sim \mathcal{N}(0, \sigma^2)$ and $i \in [1, N]$). After each agent has spend one episode in the environment a weighted sum over each agents policy network's parameters and gained reward is being created. This weighted sum of parameter vectors becomes the new base parameters. 

Mathematically, evolution strategies uses finite differences along a few random directions at each optimization step to estimate the gradients of the reward function $R$ with respect to the parameter vector $\bm \theta$. The estimated gradient is then used to update the policy network's parameters in a direction that increases the reward. This process is repeated until the desired level of performance is reached.


## Genetic Optimization

Inspired by evolution, genetic optimization uses a population of individuals that are slightly different from each other. These differences result from mutation, which is a fundamental property of evolution, and result in different behaviors in each agent. The difference in behavior makes some agents more successful than others. The fitness or success of an agent is represented by the fitness or reward function.

The algorithm begins with a population of candidate solutions, from which the agent with the highest fitness level advances to the next round. After selection, the fittest individual propagates its genetic traits to the next generation, with random mutations. This process is repeated until the desired fitness level is reached.

Here, we use the mutation operation during the optimization process to learn to land a rocket booster. Mutation operations act on all members of a population. The mutation operation is defined by the probability with which a parameter mutation is going to happen and the rate or strength with which the mutation acts on the parameter.


## Simulated Annealing

In 1983, Kirkpatrick et al., combined the insights of heating and cooling materials to change their physical properties with the Monte Carlo-based Metropolis-Hastings algorithm, to find approximate solutions to the traveling salesman problem. This combination led to the technique of simulated annealing. It has been shown, that with a sufficiently high initial temperature and sufficiently long cooling time, the system's minimum-energy state is reached.

In a nutshell, simulated annealing selects at each iteration a randomly created candidate solution that is close to the current one under some distance metric. The system moves to the proposed solution either if it comes with a higher reward or with a temperature-dependent probability. With decreasing temperature, the temperature-dependent probability to accept worse solutions narrows and the optimization focuses more and more on improving solutions approaching a Monte Carlo algorithm behavior.

In a previous project called [*NeuralAnnealing*](https://github.com/kaifishr/NeuralAnnealing), I implemented simulated annealing with [JAX](https://jax.readthedocs.io/en/latest/) to optimize neural networks for a classification task. I had good results with this approach, so I thought I would try using it here.


# TODOs

- Add explicit fuel constraint.
- Add dynamic landing legs.


# References

Salimans et al., 2017, [*Evolution Strategies as a Scalable Alternative to Reinforcement Learning*](https://arxiv.org/abs/1703.03864)

Yaghmaie et al., 2021, [*A Crash Course on Reinforcement Learning*](https://arxiv.org/abs/2103.04910)

[Box2D](https://box2d.org/) website

[Box2D C++](https://box2d.org/documentation/) documentation

[PyBox2D](https://github.com/pybox2d/pybox2d) on Github

[Backends](https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples/backends) for PyBox2D

PyBox2D [tutorial](https://github.com/pybox2d/cython-box2d/blob/master/docs/source/getting_started.md)

[Minimal PyBox2D examples](https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples)


# Citation

If you find this project useful, please use BibTeX to cite it as:

```bibtex
@article{fischer2022rocketlander,
  title   = "RocketLander",
  author  = "Fischer, Kai",
  journal = "GitHub repository",
  year    = "2022",
  month   = "December",
  url     = "https://github.com/kaifishr/RocketLander"
}
```


# License

MIT