"""
Data Processing Demo
===================

This demo shows a comprehensive data processing pipeline using SpikeInterface, including:
1. Loading and preprocessing data
2. Computing quality metrics
3. Automated curation using pre-trained models
4. Visualization of results

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.curation as sc
import spikeinterface.widgets as sw
import spikeinterface.sortingcomponents as sc_sort
from spikeinterface.sortingcomponents.peak_detection import DetectPeakLocallyExclusive
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import (
    compute_snrs,
    compute_firing_rates,
    compute_isi_violations,
    calculate_pc_metrics,
    compute_quality_metrics,
)
from probeinterface import Probe

def main():
    ##############################################################################
    # Download and load example data
    # -----------------------------
    #
    # 

    print("Loading Blackrock dataset...")
    # Use SpikeInterface's Blackrock extractor which handles the data properly
    recording = se.BlackrockRecordingExtractor('/Users/nielsnovotny/Downloads/Hub1-datafile001.ns6')
    channel_ids = recording.get_channel_ids()
    print(f"Channel IDs: {channel_ids}")
    traces = recording.get_traces(channel_ids=[channel_ids[0]])
    print(f"Recording shape: {traces.shape} (using channel_ids=[{channel_ids[0]}])")
    if traces.ndim == 1:
        traces = traces.reshape(-1, 1)
        print(f"Reshaped to 2D: {traces.shape}")
    # Use NumpyRecording to ensure compatibility with single-channel data
    from spikeinterface.core import NumpyRecording
    recording = NumpyRecording(
        traces,
        sampling_frequency=recording.get_sampling_frequency(),
        channel_ids=[str(channel_ids[0])]
    )
    # Add gain information for scaling (assuming 1.0 gain for demo purposes)
    recording.set_property("gain_to_uV", [1.0])
    recording.set_property("offset_to_uV", [0.0])
    print(f"Created NumpyRecording: {recording}")
    print(f"Sampling frequency: {recording.get_sampling_frequency()} Hz")
    print(f"Number of channels: {recording.get_num_channels()}")
    print(f"Duration: {recording.get_total_duration()} seconds")

    # Create and attach a probe for the recording
    # Create a probe with 1 shank and 1 channel
    probe = Probe(ndim=2)
    probe.set_contacts(
        positions=np.array([[0, 0]]),  # Single contact at origin
        shapes='circle',
        shape_params={'radius': 5},  # 5 micron radius
        shank_ids=[0]  # All contacts on shank 0
    )
    probe.set_device_channel_indices([0])  # Map channel 0 to contact 0
    
    # Attach the probe to the recording
    recording.set_probe(probe, in_place=True)
    print(f"Probe attached: {recording.get_probe()}")

    # Detect spikes using threshold detection
    # Prepare arguments for detect_peaks
    traces = recording.get_traces()
    peak_sign = "neg"
    abs_thresholds = np.array([5])  # threshold in standard deviations, one per channel
    exclude_sweep_size = int(0.1 * recording.get_sampling_frequency() / 1000)  # 0.1 ms in samples
    neighbours_mask = np.ones((1, 1), dtype=bool)  # For single channel, allow self-neighbor

    peaks = DetectPeakLocallyExclusive.detect_peaks(
        traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask
    )
    print(f"Detected {len(peaks[0])} peaks")
    
    # peaks is a tuple: (sample_indices, channel_indices)
    peaks_struct = np.zeros(
        len(peaks[0]),
        dtype=[("sample_index", "int64"), ("channel_index", "int64"), ("segment_index", "int64")]
    )
    peaks_struct["sample_index"] = peaks[0]
    peaks_struct["channel_index"] = peaks[1]
    peaks_struct["segment_index"] = 0  # All zeros for single segment

    # Convert peaks to sorting
    from spikeinterface.core import NumpySorting
    sorting = NumpySorting.from_peaks(
        peaks_struct,
        sampling_frequency=recording.get_sampling_frequency(),
        unit_ids=["unit_0"]  # Single unit for all peaks
    )
    print(f"Created sorting with {len(sorting.get_unit_ids())} units")

    ##############################################################################
    # Create SortingAnalyzer and compute necessary extensions
    # -----------------------------------------------------
    #
    # The SortingAnalyzer is a powerful object that allows us to compute and store various
    # metrics and features of our sorted data.

    print("\nCreating SortingAnalyzer and computing extensions...")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory"  # Store in memory for faster access
    )

    # Compute necessary extensions for quality metrics
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=600, seed=2205)
    analyzer.compute("waveforms", ms_before=1.3, ms_after=2.6, n_jobs=2)
    analyzer.compute("templates", operators=["average", "median", "std"])
    analyzer.compute("noise_levels")
    analyzer.compute("principal_components", n_components=3, mode="by_channel_global", whiten=True)
    analyzer.compute("spike_locations")
    analyzer.compute("spike_amplitudes")
    analyzer.compute("correlograms")
    analyzer.compute("quality_metrics")
    # Compute all required template metrics for model-based curation
    required_metrics = [
        'half_width', 'num_negative_peaks', 'peak_to_valley', 'num_positive_peaks',
        'peak_trough_ratio', 'recovery_slope', 'repolarization_slope'
    ]
    analyzer.compute('template_metrics', metric_names=required_metrics)

    print(f"Analyzer: {analyzer}")

    ##############################################################################
    # Compute and display quality metrics
    # ----------------------------------
    #
    # We'll compute various quality metrics to assess the quality of our sorted units.

    print("\nComputing quality metrics...")
    # Compute individual metrics
    firing_rates = compute_firing_rates(analyzer)
    snrs = compute_snrs(analyzer)
    # Note: We're not computing individual ISI violations anymore since we're using presence_ratio instead

    # Compute comprehensive metrics
    metrics = compute_quality_metrics(
        analyzer,
        metric_names=[
            "firing_rate",
            "snr",
            "amplitude_cutoff",
            "firing_range",
            "isolation_distance",
            "d_prime",
        ],
    )

    print("\nQuality metrics summary:")
    print(metrics)

    ##############################################################################
    # Automated curation using pre-trained models
    # ------------------------------------------
    #
    # We'll use a pre-trained model to automatically curate our units. We'll use the toy tetrode
    # model for this demo.

    print("\nLoading pre-trained model for automated curation...")
    model, model_info = sc.load_model(
        repo_id="SpikeInterface/toy_tetrode_model",
        trusted=['numpy.dtype']
    )

    # Apply the model to our data
    print("Applying model to label units...")
    labels = sc.auto_label_units(
        sorting_analyzer=analyzer,
        repo_id="SpikeInterface/toy_tetrode_model",
        trusted=['numpy.dtype']
    )

    print("\nModel predictions:")
    print(labels)

    ##############################################################################
    # Visualize results
    # ----------------
    #
    # Let's visualize some of our results to better understand the quality of our units.

    # Plot quality metrics distributions
    plt.figure(figsize=(15, 10))

    # Plot firing rates
    plt.subplot(2, 2, 1)
    plt.hist(metrics['firing_rate'], bins=20)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Count')
    plt.title('Firing Rate Distribution')

    # Plot SNR distribution
    plt.subplot(2, 2, 2)
    plt.hist(metrics['snr'], bins=20)
    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.title('SNR Distribution')

    # Plot firing range distribution
    plt.subplot(2, 2, 3)
    plt.hist(metrics['firing_range'], bins=20)
    plt.xlabel('Firing Range')
    plt.ylabel('Count')
    plt.title('Firing Range Distribution')

    # Plot isolation distances
    plt.subplot(2, 2, 4)
    isolation_distances = metrics['isolation_distance'].dropna()  # Remove NaN values
    if len(isolation_distances) > 0:
        plt.hist(isolation_distances, bins=20)
        plt.xlabel('Isolation Distance')
        plt.ylabel('Count')
        plt.title('Isolation Distance Distribution')
    else:
        plt.text(0.5, 0.5, 'No isolation distance data\n(requires multi-channel)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Isolation Distance Distribution')

    plt.tight_layout()
    plt.show()

    # Plot templates for some units
    # Get units with highest and lowest SNR as examples
    good_unit = metrics['snr'].idxmax()
    bad_unit = metrics['snr'].idxmin()

    plt.figure(figsize=(15, 5))
    sw.plot_unit_templates(analyzer, unit_ids=[good_unit, bad_unit])
    plt.suptitle('Unit Templates (High SNR vs Low SNR)')
    plt.show()

if __name__ == '__main__':
    main() 