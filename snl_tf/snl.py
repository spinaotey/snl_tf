import sys
import numpy as np
from snl_tf.train import ConditionalTrainer
from snl_tf import mcmc,simulators 

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

    def learn_likelihood(self, obs_xs, model, trainer, sess, n_samples, n_rounds, train_on_all=True, 
                         max_iterations=1000, batch_size=100,early_stopping=20, check_every_N=5, p_val=0.2,
                         thin=10, save_models=False,model_sufix="model",logger=sys.stdout,
                         rng=np.random):
        """
        :param obs_xs: the observed data
        :param model: the model to train
        :param sess: tensorflow session where the model is run
        :param n_samples: number of simulated samples per round
        :param n_rounds: number of rounds
        :param train_on_all: whether to train on all simulated samples or just on the latest batch
        :param max_iterations: maximum number of iterations for training.
        :param batch_size: batch size in each training iteration.
        :param early_stopping: number of iterations for early stopping criteria.
        :param check_every_N: check every N iterations if model has improved and saves if so.
        :param p_val: percentage of training data randomly selected to be used for validation in each round.
        :param thin: number of samples to thin the chain
        :param logger: logs messages
        :param rng: random number generator
        :return: the trained model
        """

        self.all_ps = []
        self.all_xs = []
        self.all_models = []
        
        if batch_size == 'all':
            use_all = True
        else:
            use_all = False
            

        log_posterior = lambda t: l_posterior_f(self.prior,model,sess,t,obs_xs)
        sampler = mcmc.SliceSampler(self.prior.gen(), log_posterior, thin=thin)
        
        try:
            aux = np.load(model_sufix+"data.npz")
            self.all_ps.append(aux['ps'])
            self.all_xs.append(aux['xs'])
            rounds = int(aux['ps'].shape[0]/n_samples)
            no_file = False
        except:
            rounds = 0
            no_file = True

        for i in range(rounds,n_rounds):

            logger.write('Learning likelihood, round {0}\n'.format(i + 1))

            if i == 0:
                # sample from prior in first round
                proposal = self.prior
            else:
                # MCMC sample posterior in every other round
                logger.write('burning-in MCMC chain...\n')
                sampler.gen(int(max(200 / thin, 1)), logger=logger, rng=rng)  # burn in
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
            if use_all:
                batch_size = ps.shape[0]-int(p_val*ps.shape[0])
            
            if save_models:
                trainer.train(sess,(ps,xs),max_iterations=max_iterations, batch_size=batch_size,p_val=p_val,
                              early_stopping=early_stopping, check_every_N=check_every_N, 
                              saver_name=model_sufix+str(i),show_log=True)
                np.savez(model_sufix+'data.npz',ps=np.concatenate(self.all_ps),xs=np.concatenate(self.all_xs))
            else:
                trainer.train(sess,(ps,xs),max_iterations=max_iterations, batch_size=batch_size,p_val=p_val,
                              early_stopping=early_stopping, check_every_N=check_every_N,show_log=True)
            logger.write('training done\n')

def l_posterior_f(prior,model,sess,t,obs_xs):
    """
    Computes log posterior probability of paramters `t` for a tensorflow
    model given observed data points `obs_xs`.
    :param prior: distribution to evaluate prior parameter density
    :param model: the model to evaluate the posterior
    :param sess: tensorflow session where the model is run
    :param obs_xs: observed data to evaluate posterior
    :return: loglikelihood of the model given observed values at parameter t
    """
    vt = np.repeat([t],obs_xs.shape[0],axis=0)
    return model.eval([vt, obs_xs],sess).sum() + prior.eval(t)