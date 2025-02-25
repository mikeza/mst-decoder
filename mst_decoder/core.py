
'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

References
----------
.. [1] Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
       (2016). Rapid classification of hippocampal replay content for
       real-time applications. Journal of Neurophysiology 116, 2221-2235.

'''
from functools import wraps
from logging import getLogger

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm

logger = getLogger(__name__)

def predict_state(data, initial_conditions=None, state_transition=None,
                  likelihood_function=None, likelihood_kwargs={}):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable

    Parameters
    ----------
    data : array_like, shape=(n_signals, n_time, ...)
    initial_conditions : array_like (n_states, n_parameters)
    state_transition : array_like (n_states, n_parameters, n_parameters)
    likelihood_function : function
    likelihood_kwargs: dict, optional
        Additional arguments to the likelihood function
        besides the data

    Returns
    -------
    posterior_over_time : array, shape=(n_time_points, n_states,
                                        n_parameters)

    '''
    n_time_points = data.shape[1]
    shape = (n_time_points, *initial_conditions.shape)
    posterior = np.zeros(shape)
    likelihood = np.zeros(shape)
    prior = np.zeros(shape)

    current_posterior = initial_conditions.copy()

    for time_ind in np.arange(n_time_points):
        prior[time_ind] = _get_prior(current_posterior, state_transition)
        likelihood[time_ind] = likelihood_function(
            data[:, time_ind, ...], **likelihood_kwargs)
        posterior[time_ind] = _update_posterior(
            prior[time_ind], likelihood[time_ind])
        current_posterior = posterior[time_ind].copy()

    return {'posterior_density': posterior,
            'likelihood': likelihood,
            'prior': prior}


def _update_posterior(prior, likelihood):
    '''The posterior density given the prior state weighted by the
    observed instantaneous likelihood
    '''
    return normalize_to_probability(prior * likelihood)


def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)


def _get_prior(posterior, state_transition):
    '''The prior given the current posterior density and a transition
    matrix indicating the state at the next time step.
    '''
    if state_transition is None:
        return posterior

    return np.matmul(
        state_transition, posterior[..., np.newaxis]).squeeze()


def scaled_likelihood(log_likelihood_func):
    '''Converts a log likelihood to a scaled likelihood with its max value at
    1.

    Used primarily to keep the likelihood numerically stable because more
    observations at a time point will lead to a smaller overall likelihood
    and this can exceed the floating point accuarcy of a machine.

    Parameters
    ----------
    log_likelihood_func : function

    Returns
    -------
    scaled_likelihood : function

    '''
    @wraps(log_likelihood_func)
    def decorated_function(*args, **kwargs):
        log_likelihood = log_likelihood_func(*args, **kwargs)
        return np.exp(log_likelihood - np.max(log_likelihood))

    return decorated_function


@scaled_likelihood
def combined_likelihood(data, log_likelihood_function=None,
                        likelihood_kwargs={}):
    '''Applies likelihood function to each signal and returns their product

    If there isn't a column dimension, just returns the likelihood.

    Parameters
    ----------
    data : array_like, shape=(n_signals, ...)
    log_likelihood_function : function
        Log Likelihood function to be applied to each signal.
        The likelihood function must take data as its first argument.
        All other arguments for the likelihood should be passed
        via `likelihood_kwargs`
    likelihood_kwargs : dict
        Keyword arguments for the likelihood function

    Returns
    -------
    likelihood : array_like, shape=(n_parameters * n_states,)

    '''
    try:
        return np.sum(
            log_likelihood_function(data, **likelihood_kwargs),
            axis=0).squeeze()
    except ValueError:
        return log_likelihood_function(data, **likelihood_kwargs).squeeze()


def empirical_movement_transition_matrix(place, place_bin_edges,
                                         sequence_compression_factor=16,
                                         is_condition=None):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `sequence_compression_factor`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    place : array_like, shape (n_time,)
        Linearized position of the animal over time
    place_bin_edges : array_like, shape (n_bins,)
    sequence_compression_factor : int, optional
        How much the movement is sped-up during a replay event
    is_condition : array_like, shape (n_time,)
        Boolean indicator for an experimental condition.
    Returns
    -------
    empirical_movement_transition_matrix : array_like,
                                           shape=(n_bin_edges-1,
                                           n_bin_edges-1)

    '''
    place = np.array(place)
    if is_condition is None:
        is_condition = np.ones_like(place, dtype=bool)

    place = np.stack((place[1:], place[:-1]))
    place = place[:, is_condition[1:]]

    movement_bins, _, _ = np.histogram2d(place[0], place[1],
                                         bins=(place_bin_edges[0],
                                               place_bin_edges[0]),
                                         normed=False)

    smoothed_movement_bins_probability = gaussian_filter(
        _normalize_row_probability(
            _fix_zero_bins(movement_bins)), sigma=0.5)
    return np.linalg.matrix_power(
        smoothed_movement_bins_probability,
        sequence_compression_factor)


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    return x / x.sum(axis=1, keepdims=True)


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    movement_bins[:, movement_bins.sum(axis=0) == 0] = 1
    return movement_bins


def linearized_bin_grid(bin_centers):
    '''Given the outer-points of bins as a n_dims length list of arrays of shape (n_bins_dim,)
     find their centers as a paired grid of shape (prod(n_bins_dim), n_dims)
    '''
    grid_bin = np.meshgrid(*bin_centers)
    return np.vstack([np.ravel(a) for a in grid_bin]).transpose()

def bin_centers(bin_edges):
    return [dim_edges[:-1] + np.diff(dim_edges) / 2 for dim_edges in bin_edges]

def uniform_initial_conditions(tuning_bin_centers):
    '''
    '''
    return normalize_to_probability(np.ones((tuning_bin_centers.shape[0])))
