'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

References
----------
.. [1] Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
       (2016). Rapid classification of hippocampal replay content for
       real-time applications. Journal of Neurophysiology 116, 2221-2235.

'''

import numpy as np
from numba import jit
from functools import partial


@jit(nopython=True)
def _normal_pdf(x, mean=0, std_deviation=1):
    '''Evaluate the normal probability density function at specified points.

    Unlike the `scipy.norm.pdf`, this function is not general and does not
    do any sanity checking of the inputs. As a result it is a much faster
    function, but you should be sure of your inputs before using.

    This function only computes the one-dimensional pdf.

    Parameters
    ----------
    x : array_like
        The normal probability function will be evaluated
    mean : float or array_like, optional
    std_deviation : float or array_like

    Returns
    -------
    probability_density
        The normal probability density function evaluated at `x`

    '''
    z = (x - mean) / std_deviation
    return np.exp(-0.5 * z ** 2) / (np.sqrt(2.0 * np.pi) * std_deviation)


def poisson_mark_likelihood(marks, joint_mark_intensity_functions=None,
                            ground_process_intensity=None,
                            time_bin_size=1):
    '''Probability of parameters given spiking indicator at a particular
    time and associated marks.

    Parameters
    ----------
    marks : array, shape (n_signals, n_marks)
    joint_mark_intensity : function
        Instantaneous probability of observing a spike given mark vector
        from data. The parameters for this function should already be set,
        before it is passed to `poisson_mark_likelihood`.
    ground_process_intensity : array, shape (n_signals, 
                                             n_tuning_bins)
        Probability of observing a spike regardless of marks.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_likelihood : array_like, shape (n_signals, n_tuning_bins)

    '''
    probability_no_spike = np.exp(-ground_process_intensity * time_bin_size)
    joint_mark_intensity = np.array(
         [jmi(signal_marks) for signal_marks, jmi
         in zip(marks, joint_mark_intensity_functions)])
    return joint_mark_intensity * probability_no_spike


def evaluate_mark_space(test_marks, training_marks=None,
                        mark_std_deviation=20):
    '''Evaluate the multivariate Gaussian kernel for the mark space
    given training marks.

    For each mark in the training data (`training_marks`), a univariate
    Gaussian is placed with its mean at the value of each mark with
    standard deviation `mark_std_deviation`. The product of the Gaussians
    along the mark dimension yields a multivariate Gaussian kernel
    evaluated at each training spike with a diagonal coviarance matrix.

    Parameters
    ----------
    test_marks : array, shape (n_marks,)
        The marks to be evaluated
    training_marks : array, shape (n_training_spikes, n_marks)
        The marks for each spike when the animal is moving
    mark_std_deviation : float, optional
        The standard deviation of the Gaussian kernel in millivolts

    Returns
    -------
    mark_space_estimator : array, shape (n_training_spikes,)

    '''
    return np.nanprod(
        _normal_pdf(test_marks, mean=training_marks,
                    std_deviation=mark_std_deviation), axis=1)


def joint_mark_intensity(marks, training_marks=None,
                         mark_std_deviation=None,
                         tuning_curve=None, stimulus_occupancy=None):
    '''Evaluate the multivariate density function of the marks and place
    field for each signal

    Parameters
    ----------
    marks : array_like, shape (n_marks,)
        Marks to be evaulated.
    tuning_curve : array_like, shape (n_tuning_bins, n_training_spikes)
        Response of the signal to stimulus.
    stimulus_occupancy : array_like, shape (n_tuning_bins,)
        How often the animal is at a stimulus
    training_marks : array_like, shape (n_training_spikes, n_marks)
        The marks for each spike when the animal is moving
    mark_std_deviation : float, optional
        The standard deviation of the Gaussian kernel in millivolts

    Returns
    -------
    joint_mark_intensity : array_like, shape (n_tuning_bins,)

    '''
    if np.any(~np.isnan(marks)):
        # If there is a spike, evaluate
        mark_space = evaluate_mark_space(
            marks, training_marks=training_marks,
            mark_std_deviation=mark_std_deviation)
        return np.dot(tuning_curve, mark_space) / stimulus_occupancy
    else:
        return np.ones_like(stimulus_occupancy)


def build_joint_mark_intensity(stimulus, training_marks, tuning_bin_centers,
                               tuning_std_deviation, mark_std_deviation):
    '''Make a joint mark intensity function with precalculated quauntities
    (`training_marks`, `tuning_curve`, 'stimulus_occupancy') preset.

    The new function only needs to be passed marks to be evaluated because
    the other quantities have already been set. See functools.partial for
    further explanation of partial functions.

    Parameters
    ----------
    stimulus : ndarray, shape (n_time, n_stimulus_dims)
        Stimulus to decode over time
    training_marks : ndarray, shape (n_time, n_marks)
        The marks over time
    tuning_bin_centers : array, shape (n_stimulus_dims, n_stimulus_bins)
        The centers of bins for stimulus
    tuning_std_deviation : float
    mark_std_deviation : float

    Returns
    -------
    joint_mark_intensity : partial function

    '''
    is_spike = np.any(~np.isnan(training_marks), axis=1)
    stimulus_occupancy = estimate_stimulus_occupancy(
        stimulus, tuning_bin_centers, tuning_std_deviation)
    tuning_curve = estimate_tuning_curve(
        stimulus, is_spike, tuning_bin_centers, tuning_std_deviation)

    return partial(
        joint_mark_intensity,
        training_marks=training_marks[is_spike, :],
        mark_std_deviation=mark_std_deviation,
        stimulus_occupancy=stimulus_occupancy,
        tuning_curve=tuning_curve
    )


def estimate_tuning_curve(stimulus, is_spike, tuning_bin_centers,
                         tuning_std_deviation=1):
    '''Non-parametric estimate of the neuron receptive field with respect
    to a stimulus

    Puts a Gaussian with a mean at the stimulus associated with the animal
    at each spike. The product of the Gaussians along the stimulus dimension
    yields a multivariate Gaussian with a diagonal covariate matrix

    Parameters
    ----------
    stimulus : narray, shape (n_time, n_stimulus_dims)
        n-dimensional stimulus over time
    is_spike : array, shape (n_time,)
        Boolean array with True indicating spike at that time.
    tuning_bin_centers : array_like, shape (n_tuning_bins, n_stimulus_dims)
        Evaluate the Gaussian at these values
    tuning_std_deviation : float, optional
        Standard deviation of the Gaussian kernel

    Returns
    -------
    tuning_curve_estimator : array, shape (n_tuning_bins, n_spikes)

    '''
    return _normal_pdf(
        tuning_bin_centers[:, np.newaxis], mean=stimulus[is_spike],
        std_deviation=tuning_std_deviation).prod(axis=-1)


def estimate_ground_process_intensity(stimulus, marks, tuning_bin_centers,
                                      tuning_std_deviation):
    '''The probability of observing a spike regardless of mark.

    Marginalizes the joint mark intensity over the mark space.

    Parameters
    ----------
    stimulus : narray, shape (n_time, n_stimulus_dims)
        n-dimensional stimulus over time
    marks : array, shape (n_time, n_marks)
    tuning_bin_centers : array, shape (n_tuning_bins,)
    tuning_std_deviation : float

    Returns
    -------
    ground_process_intensity : array, shape (n_tuning_bins,)

    '''
    is_spike = np.any(~np.isnan(marks), axis=1)
    tuning_curve = estimate_tuning_curve(
        stimulus, is_spike, tuning_bin_centers, tuning_std_deviation)
    stimulus_occupancy = estimate_stimulus_occupancy(
        stimulus, tuning_bin_centers, tuning_std_deviation)
    return tuning_curve.sum(axis=1) / stimulus_occupancy


def estimate_stimulus_occupancy(stimulus, tuning_bin_centers,
                             tuning_std_deviation=1):
    '''How often the animal is at a stimulus

    Denominator in equation #12 and #13 of [1]

    Parameters
    ----------
    stimulus : narray, shape (n_time, n_stimulus_dims)
        n-dimensional stimulus over time
    tuning_bin_centers : array_like, shape (n_tuning_bins, n_stimulus_dims)
    tuning_std_deviation : float, optional

    Returns
    -------
    stimulus_occupancy : array, shape (n_tuning_bins,)
        How often the animal is at a stimulus
    '''
    return _normal_pdf(tuning_bin_centers[:, np.newaxis, :], mean=stimulus,
                std_deviation=tuning_std_deviation).prod(axis=-1).sum(axis=1)

    # A new axis is introduced to find the difference for each stimulus value
    # The product of the single gaussians yields a multivariate gaussian over all dimensions
    # The sum yields a result over all stimulus values seen

def estimate_marginalized_joint_mark_intensity(
    mark_bin_centers, training_marks, tuning_curve, stimulus_occupancy,
        mark_std_deviation):
    '''

    Parameters
    ----------
    mark_bin_centers : array, shape (n_mark_bins,)
    training_marks : array, shape (n_spikes, n_marks)
    tuning_curve : array, shape (n_tuning_bins, n_spikes)
    stimulus_occupancy : array, shape (n_tuning_bins,)
    mark_std_deviation : float

    Returns
    -------
    marginalized_joint_mark_intensity : array, shape (n_tuning_bins,
                                                      n_mark_bins, n_marks)

    '''

    mark_at_spike = _normal_pdf(
        mark_bin_centers[:, np.newaxis, np.newaxis], training_marks,
        mark_std_deviation)
    return (np.dot(tuning_curve, mark_at_spike) /
            stimulus_occupancy[:, np.newaxis, np.newaxis])
