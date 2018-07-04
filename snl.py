import sys
import numpy as np
from scipy.misc import logsumexp
from train import ConditionalTrainer
import mcmc

class SequentialNeuralLikelihood:
    """
    Trains a likelihood model using posterior MCMC sampling to guide simulations.
    """

    def __init__(self, prior, sim_model):

        self.prior = prior
        self.sim_model = sim_model

        self.all_ps = None
        self.all_xs = None
        self.all_models = None

    def learn_likelihood(self, obs_xs, model, sess, n_samples, n_rounds, train_on_all=True, thin=10, 
                         save_models=False, model_sufix="model",logger=sys.stdout, rng=np.random):
        """
        :param obs_xs: the observed data
        :param model: the model to train
        :param sess: tensorflow session where the model is run
        :param n_samples: number of simulated samples per round
        :param n_rounds: number of rounds
        :param train_on_all: whether to train on all simulated samples or just on the latest batch
        :param thin: number of samples to thin the chain
        :param logger: logs messages
        :param rng: random number generator
        :return: the trained model
        """

        self.all_ps = []
        self.all_xs = []
        self.all_models = []

        log_posterior = lambda t: model.eval([t, obs_xs],sess).flatten() + self.prior.eval(t)
        sampler = mcmc.SliceSampler(self.prior.gen(), log_posterior, thin=thin)
        trainer = ConditionalTrainer(model)

        for i in range(n_rounds):

            logger.write('Learning likelihood, round {0}\n'.format(i + 1))

            if i == 0:
                # sample from prior in first round
                proposal = self.prior
            else:
                # MCMC sample posterior in every other round
                logger.write('burning-in MCMC chain...\n')
                sampler.gen(max(200 / thin, 1), logger=logger, rng=rng)  # burn in
                logger.write('burning-in done...\n')
                proposal = sampler

            # run new batch of simulations
            logger.write('simulating data... ')
            ps, xs = simulators.sim_data(proposal.gen, self.sim_model, n_samples, rng=rng)
            logger.write('done\n')
            self.all_ps.append(ps)
            self.all_xs.append(xs)

            if train_on_all:
                ps = np.concatenate(self.all_ps)
                xs = np.concatenate(self.all_xs)

            N = ps.shape[0]
            monitor_every = min(10 ** 5 / float(N), 1.0)

            # retrain likelihood model
            logger.write('training model...\n')
            if save_models:
                trainer.train(sess,(ps,xs),saver_name=model_sufix+str(i),show_log=True)
            else:
                trainer.train(sess,(ps,xs),show_log=True)
            logger.write('training done\n')