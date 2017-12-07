'''Tools for simulating place field spiking'''
import numpy as np
from scipy.stats import norm, multivariate_normal
from .utils import atleast_2d


def simulate_time(n_samples, sampling_frequency):
    return np.arange(n_samples) / sampling_frequency


def simulate_linear_distance(time, track_height):
    return ((track_height / 2) * np.sin(2 * np.pi * time - (np.pi) / 2) +
            (track_height / 2))

def simulate_nd_stimulus(time, bounds):
    return ((track_height / 2) * np.sin(2 * np.pi * time - (np.pi) / 2) +
            (track_height / 2))

def simulate_circular_cartesian(time, radius=50, rate=5):
    ''' Generates circular movement over time returning array of x y positions shape (n_time, 2)
    Rate is seconds for a full rotation
    '''
    rotation_frac = time % rate
    theta = rotation_frac * 2*np.pi / rate
    return radius * np.array([np.cos(theta), np.sin(theta)]).transpose()

def generate_gridded_tuning_curve_means(k, stimulus):
    ''' Generates tuning curve means for k neurons for n-dimensional data inferred from 
    the stimulus data provided
    '''
    stim = atleast_2d(stimulus)
    n_dims = stim.shape[1]
    k_per_dim = np.floor(np.power(k, 1 / n_dims)) + 1
    mins = np.min(stim, axis=0)
    print(mins)
    maxs = np.max(stim, axis=0)
    curve_edges = [np.linspace(mins[i], maxs[i], k_per_dim) for i in range(n_dims)]
    curve_centers = [dim_edges[:-1] + np.diff(dim_edges) / 2 for dim_edges in curve_edges]
    grid_centers = np.meshgrid(*curve_centers)
    return np.vstack([np.ravel(a) for a in grid_centers]).transpose()

def simulate_poisson_spikes(rate, sampling_frequency):
    return 1.0 * (np.random.poisson(rate / sampling_frequency) > 0)

def create_tuning_curve(tuning_curve_mean, stimulus, sampling_frequency, 
    tuning_curve_std_deviation=12.5,  max_firing_rate=10, baseline_firing_rate=2):
    tuned_firing_rate = norm(
        tuning_curve_mean, tuning_curve_std_deviation).pdf(atleast_2d(stimulus)).prod(axis=-1)
    tuned_firing_rate /= tuned_firing_rate.max()
    return baseline_firing_rate + max_firing_rate * tuned_firing_rate

def generate_marks(spikes, mark_mean, mark_std_deviation, n_marks=4):
    '''Generate a tuning curve with an associated mark'''
    spikes[spikes == 0] = np.nan
    marks = multivariate_normal(
        mean=[mark_mean] * n_marks,
        cov=[mark_std_deviation] * n_marks).rvs(size=(spikes.size,))
    return marks * spikes[:, np.newaxis]


def simulate_multiunit(
    tuning_curve_means, mark_means, stimulus, sampling_frequency,
        mark_std_deviation=20, n_marks=4, **kwargs):
    '''Simulate a single tetrode assuming each tetrode picks up several
    neurons with different tuning curves with distinguishing marks.'''
    unit = []
    for tuning_curve_mean, mark_mean in zip(tuning_curve_means, mark_means):
        rate = create_tuning_curve(
            tuning_curve_mean, stimulus, sampling_frequency, **kwargs)
        spikes = simulate_poisson_spikes(rate, sampling_frequency)
        marks = generate_marks(
            spikes, mark_mean, mark_std_deviation, n_marks=n_marks)
        unit.append(marks)

    return np.nanmean(np.stack(unit, axis=0), axis=0)
