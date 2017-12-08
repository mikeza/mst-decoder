
from logging import getLogger

import numpy as np

# Dependencies to remove
import xarray as xr

# Dependencies for debugging
import memory_profiler
from time import perf_counter
import threading

logger = getLogger(__name__)

from .clusterless import (build_joint_mark_intensity,
                          estimate_ground_process_intensity,
                          estimate_marginalized_joint_mark_intensity,
                          poisson_mark_likelihood)
from .core import (combined_likelihood, empirical_movement_transition_matrix,
                   get_grid_bin_centers, predict_state, uniform_initial_conditions)
from .utils import (atleast_2d)

class ClusterlessDecoder(object):
    ''' 

    Attributes
    ----------
    stimulus : array, shape (n_time, n_stimulus_dims)
        Stimulus of interest 
    spike_marks : array, shape (n_signals, n_time, n_marks)
        Spike marks recorded from a multiple signals
        If a spike does not occur, rows are np.nan
    n_stimulus_bins : array, shape (n_stimulus_dims,), optional
        Number of bins per stimulus dimension
    mark_std_deviation : int, optional
        Standard deviation for mark space estimation
    tuning_std_deviation : array, shape (n_stimulus_dims,), optional
        Standard deviation for estimation of tuning curves for each stimulus dimension

    '''

    def __init__(self, stimulus, spike_marks, 
                 n_stimulus_bins=31,
                 mark_std_deviation=20,
                 tuning_std_deviation=None,
                 time_bin_size=1):

        self.stimulus = atleast_2d(np.array(stimulus))
        self.spike_marks = np.array(spike_marks)
        self.n_stimulus_bins = n_stimulus_bins
        self.mark_std_deviation = mark_std_deviation
        self.tuning_std_deviation = tuning_std_deviation
        self.time_bin_size = time_bin_size

    def fit(self):

        self.n_stimulus_dims = self.stimulus.shape[1]

        if np.isscalar(self.n_stimulus_bins):
            self.n_stimulus_bins = np.repeat(self.n_stimulus_bins, 
                                             self.n_stimulus_dims)

        mins = self.stimulus.min(axis=0)
        maxs = self.stimulus.max(axis=0)
        self.tuning_bin_edges = [np.linspace(mins[i], maxs[i], self.n_stimulus_bins[i] + 1) 
                                for i in range(self.n_stimulus_dims)]

        if self.tuning_std_deviation is None:
            self.tuning_std_deviation = (maxs - mins) / self.n_stimulus_bins
        elif np.isscalar(self.tuning_std_deviation):
            self.tuning_std_deviation = np.repeat(self.tuning_std_deviation, 
                                                  self.n_stimulus_dims)

        self.tuning_bin_centers = get_grid_bin_centers(self.tuning_bin_edges)

        self.initial_conditions = uniform_initial_conditions(self.tuning_bin_centers)

        self.state_transition_matrix = None

        joint_mark_intensity_funcs = []
        ground_process_intensities = []

        for signal_marks in self.spike_marks:

            jmi = build_joint_mark_intensity(
                self.stimulus,
                signal_marks,
                self.tuning_bin_centers,
                self.tuning_std_deviation,
                self.mark_std_deviation)

            joint_mark_intensity_funcs.append(jmi)

            gpi = estimate_ground_process_intensity(
                self.stimulus,
                signal_marks,
                self.tuning_bin_centers,
                self.tuning_std_deviation)

            ground_process_intensities.append(gpi)

        likelihood_kwargs = dict(
            joint_mark_intensity_functions=joint_mark_intensity_funcs,
            ground_process_intensity=ground_process_intensities,
            time_bin_size=self.time_bin_size)

        self._combined_likelihood_kwargs = dict(
            likelihood_function=poisson_mark_likelihood,
            likelihood_kwargs=likelihood_kwargs)

        return self

    def predict(self, spike_marks, time=None):
        '''Predicts the stimulus from spike_marks.

        Parameters
        ----------
        spike_marks : array, shape (n_signals, n_time, n_marks)
            If spike does not occur, the row must be marked with np.nan.
        time : array, optional, shape (n_time,)

        Returns
        -------
        predicted_state : str

        '''
        # results = predict_state(
        #     spike_marks,
        #     initial_conditions=self.initial_conditions.values,
        #     state_transition=self.state_transition_matrix.values,
        #     likelihood_function=combined_likelihood,
        #     likelihood_kwargs=self._combined_likelihood_kwargs)

        # coords = dict(
        #     time=(time if time is not None
        #           else np.arange(results['posterior_density'].shape[0])),
        #     position=self.tuning_bin_centers
        # )

        # DIMS = ['time', 'position']

        # results = xr.Dataset(
        #     {key: (DIMS, value) for key, value in results.items()},
        #     coords=coords)

        # return DecodingResults(
        #     results=results,
        #     spikes=spike_marks,
        #     confidence_threshold=self.confidence_threshold,
        # )

    def marginalized_intensities(self):
        joint_mark_intensity_functions = (
            self._combined_likelihood_kwargs['likelihood_kwargs']
            ['joint_mark_intensity_functions'])
        mark_bin_centers = np.linspace(100, 350, 200)

        marginalized_intensities = np.stack(
            [estimate_marginalized_joint_mark_intensity(
                mark_bin_centers, jmi.keywords['training_marks'],
                jmi.keywords['tuning_curve'],
                jmi.keywords['stimulus_occupancy'], self.mark_std_deviation)
             for jmi in joint_mark_intensity_functions])

        dims = ['signal', 'stimulus_bin', 'marks', 'mark_dimension']

        coords = dict(
            marks=mark_bin_centers,
            stimulus_bin=self.tuning_bin_centers[:, 0]
        )

        return xr.DataArray(marginalized_intensities, dims=dims,
                            coords=coords)

    def plot_observation_model(self):
        marginalized_intensities = (
            self.marginalized_intensities().sum('mark_dimension'))
        try:
            return marginalized_intensities.plot(
                row='signal', x='stimulus_bin', y='marks',
                robust=True)
        except ValueError:
            return marginalized_intensities.plot(
                row='signal', x='stimulus_bin', y='marks', robust=True)
