import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))

    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


class LinearDecayScheduler(object):
    """Set hyper parameters by a step-based scheduler with linear decay
    values."""

    def __init__(self, start_value, max_steps):
        """Linear decay scheduler of hyper parameter. Decay value linearly
        until 0.

        Args:
            start_value (float): start value
            max_steps (int): maximum steps
        """
#        assert max_steps > 0
        self.cur_step = 0
        self.max_steps = max_steps
        self.start_value = start_value

    def step(self, step_num=1):
        """Step step_num and fetch value according to following rule:

        return_value = start_value * (1.0 - (cur_steps / max_steps))

        Args:
            step_num (int): number of steps (default: 1)

        Returns:
            value (float): current value
        """
        assert isinstance(step_num, int) and step_num >= 1
        self.cur_step = min(self.cur_step + step_num, self.max_steps)

        value = self.start_value * (1.0 -
                                    ((self.cur_step * 1.0) / self.max_steps))

        return value