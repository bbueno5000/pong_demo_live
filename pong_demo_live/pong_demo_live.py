"""
Trains an agent with (stochastic) Policy Gradients on Pong.
"""
import cPickle 
import gym
import numpy
import sys

class Pong:
    """
    TODO: docstring
    """
    def __init__(self):
        """
        TODO: docstring
        """
        H = 200
        self.batch_size = 10
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.decay_rate = 0.99
        resume = False
        D = 80 * 80
        if resume:
            self.model = cPickle.load(open('save.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = numpy.random.randn(H,D) / numpy.sqrt(D)
            self.model['W2'] = numpy.random.randn(H) / numpy.sqrt(H)
        self.grad_buffer = {k: numpy.zeros_like(v) for k, v in self.model.iteritems()} 
        self.rmsprop_cache = {k: numpy.zeros_like(v) for k, v in self.model.iteritems()} 
   
    def __call__(self):
        """
        TODO: docstring
        """
        env = gym.make('Pong-v0')
        observation = env.reset()
        prev_x = None
        xs,hs,dlogps,drs = [], [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0
        while True:
            cur_x = self.prepro(observation)
            x = cur_x - prev_x if prev_x is not None else numpy.zeros(D)
            prev_x = cur_x
            aprob, h = self.policy_forward(x)
            action = 2 if numpy.random.uniform() < aprob else 3
            hs.append(h)
            y = 1 if action == 2 else 0
            dlogps.append(y - aprob)
            env.render()
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            drs.append(reward)
            if done:
                episode_number += 1
                epx = numpy.vstack(xs)
                eph = numpy.vstack(hs)
                epdlogp = numpy.vstack(dlogps)
                epr = numpy.vstack(drs)
                xs, hs, dlogps, drs = [], [], [], []
                discounted_epr = self.discount_rewards(epr)
                discounted_epr -= numpy.mean(discounted_epr)
                discounted_epr /= numpy.std(discounted_epr)
                epdlogp *= discounted_epr
                grad = self.policy_backward(eph, epdlogp)
                for k in self.model: self.grad_buffer[k] += grad[k]
                if episode_number % self.batch_size == 0:
                    for k, v in self.model.iteritems():
                        g = self.grad_buffer[k]
                        self.rmsprop_cache[k] = \
                            self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                        self.model[k] += self.learning_rate * g / (
                            numpy.sqrt(self.rmsprop_cache[k]) + 1e-5)
                        self.grad_buffer[k] = numpy.zeros_like(v)
                running_reward = \
                    reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode reward total was {}. running mean: {}')·format(
                    reward_sum, running_reward)
                if episode_number % 100 == 0:
                    cPickle.dump(self.model, open('save.p', 'wb'))
                reward_sum = 0
                observation = env.reset()
                prev_x = None
            if reward != 0:
                print('ep {}: game finished, reward: {}').format(episode_number, reward)

    def discount_rewards(self, r):
        """
        Take 1D float array of rewards and compute discounted reward.
        """
        discounted_r = numpy.zeros_like(r)
        running_add = 0
        for t in reversed(range(r.size)):
            if r[t] != 0:
                running_add = 0 
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_backward(self, eph, epdlogp):
        """
        Backward pass. (eph is array of intermediate hidden states).
        """
        dW2 = numpy.dot(eph.T, epdlogp).ravel()
        dh = numpy.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0
        dW1 = numpy.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    def policy_forward(self, x):
        """
        Forward propagation.
        """
        h = numpy.dot(self.model['W1'], x)
        h[h<0] = 0
        logp = numpy.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h

    def prepro(self, I):
        """
        Takes a single game frame as input.
        Lreprocesses before feeding into model
        prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector.
        """
        I = I[35:195]
        I = I[::2,::2,0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        return I.astype(numpy.float).ravel()

    def sigmoid(self, x): 
        """
        Sigmoid "squashing" function to interval [0,1].
        """
        return 1.0 / (1.0 + numpy.exp(-x))

def main(argv):
    """
    TODO: docstring
    """
    pong = Pong()
    pong()

if __name__ == '__main__':
    main(sys.argv)
