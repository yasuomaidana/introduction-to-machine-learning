from abc import ABCMeta

from agent import BaseAgent
import numpy as np


class DynaQAgent(BaseAgent, metaclass=ABCMeta):

    def __init__(self, agent_info: dict):
        super().__init__()

        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except KeyError:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")

        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, etc.
        # A simple way to implement the model is to have a dictionary of dictionaries,
        #        mapping each state to a dictionary which maps actions to (reward, next state) tuples.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {}  # model is a dictionary of dictionaries, which maps states to actions to
        # (reward, next_state) tuples

    def agent_init(self, agent_info: dict):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_init_info ( :param agent_info: ), the parameters used to initialize the agent. The dictionary contains:

            {
                num_states (int): The number of states,
                num_actions (int): The number of actions,
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """

        # First, we get the relevant information from agent_info
        # NOTE: we use np.random.RandomState(seed) to set the two different RNGs
        # for the planner and the rest of the code
        try:
            self.num_states = agent_info["num_states"]
            self.num_actions = agent_info["num_actions"]
        except KeyError:
            print("You need to pass both 'num_states' and 'num_actions' \
                   in agent_info to initialize the action-value table")
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 10)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 42))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 42))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, etc.
        # A simple way to implement the model is to have a dictionary of dictionaries,
        #        mapping each state to a dictionary which maps actions to (reward, next state) tuples.
        self.q_values = np.zeros((self.num_states, self.num_actions))
        self.actions = list(range(self.num_actions))
        self.past_action = -1
        self.past_state = -1
        self.model = {}  # model is a dictionary of dictionaries, which maps states to actions to
        # (reward, next_state) tuples

    def update_model(self, past_state: int, past_action: int, state: int, reward: float):
        """updates the model

        Args:
            past_state       (int): s
            past_action      (int): a
            state            (int): s'
            reward           (int): r
        Returns:
            Nothing
        """
        # Update the model with the (s,a,s',r) tuple (1~4 lines)

        # ----------------
        # your code here
        model_sa = self.model.get(past_state, {})
        model_sa[past_action] = state, reward
        self.model[past_state] = model_sa
        # ----------------

    def planning_step(self):
        """performs planning, i.e. indirect RL.

        Args:

        Returns:
            Nothing
        """

        # The indirect RL step:
        # - Choose a state and action from the set of experiences that are stored in the model. (~2 lines)
        # - Query the model with this state-action pair for the predicted next state and reward.(~1 line)
        # - Update the action values with this simulated experience.                            (2~4 lines)
        # - Repeat for the required number of planning steps.
        #
        # Note that the update equation is different for terminal and non-terminal transitions.
        # To differentiate between a terminal and a non-terminal next state, assume that the model stores
        # the terminal state as a dummy state like -1
        #
        # Important: remember you have a random number generator 'planning_rand_generator' as
        #     a part of the class which you need to use as self.planning_rand_generator.choice()
        #     For the sake of reproducibility and grading, *do not* use anything else like
        #     np.random.choice() for performing search control.

        # ----------------
        # your code here
        for _ in range(self.planning_steps):
            s = self.planning_rand_generator.choice(list(self.model.keys()))
            a = self.planning_rand_generator.choice(list(self.model[s].keys()))
            next_s, reward = self.model[s][a]
            self.q_values[s, a] += (self.step_size *
                                    (reward
                                     + self.gamma * (np.max(self.q_values[next_s, :]) if next_s != -1 else 0)
                                     - self.q_values[s, a]))
        # ----------------

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state):
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """

        # given the state, select the action using self.choose_action_egreedy()),
        # and save current state and action (~2 lines)
        ### self.past_state = ?
        ### self.past_action = ?

        # ----------------
        # your code here
        self.past_state = state
        self.past_action = self.choose_action_egreedy(state)
        # ----------------

        return self.past_action

    def agent_step(self, reward, state):
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """

        # - Direct-RL step (~1-3 lines)
        # - Model Update step (~1 line)
        # - `planning_step` (~1 line)
        # - Action Selection step (~1 line)
        # Save the current state and action before returning the action to be performed. (~2 lines)

        # ----------------
        # your code here
        self.q_values[self.past_state, self.past_action] += (self.step_size *
                                                             (reward
                                                              + self.gamma * (
                                                                  np.max(self.q_values[state, :]) if state != -1 else 0)
                                                              - self.q_values[self.past_state, self.past_action]))
        self.update_model(self.past_state, self.past_action, state, reward)
        self.planning_step()
        self.past_state = state
        self.past_action = self.choose_action_egreedy(state)
        # ----------------

        return self.past_action

    def agent_end(self, reward):
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # - Direct RL update with this final transition (1~2 lines)
        # - Model Update step with this final transition (~1 line)
        # - One final `planning_step` (~1 line)
        #
        # Note: the final transition needs to be handled carefully. Since there is no next state,
        #       you will have to pass a dummy state (like -1), which you will be using in the planning_step() to
        #       differentiate between updates with usual terminal and non-terminal transitions.

        # ----------------
        # your code here
        self.q_values[self.past_state, self.past_action] += (self.step_size *
                                                             (reward
                                                              - self.q_values[self.past_state, self.past_action]))
        self.update_model(self.past_state, self.past_action, -1, reward)
        self.planning_step()
        # ----------------