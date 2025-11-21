"""
Comprehensive PSG Preprocessing Script
Performs channel resampling, artifact removal, bandpass filtering, and interpolation
for any PSG dataset (EDF format).

Author: Sleep Staging Team
Date: 2024
"""

import os
import numpy as np
import mne
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from typing import List, Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class PSGPreprocessor:
    """
    Comprehensive PSG preprocessing pipeline for sleep staging datasets.
    
    Features:
    - Channel resampling to target frequency
    - Artifact detection and removal (multiple methods)
    - Bandpass filtering (channel-specific frequency bands)
    - Bad channel detection and interpolation
    - Notch filtering for line noise
    """
    
    def __init__(
        self,
        target_sfreq: float = 100.0,
        notch_freq: Optional[float] = 50.0,  # 50Hz in Europe, 60Hz in US
        notch_quality: float = 30.0,
        filter_method: str = 'fir',
        filter_length: str = 'auto',
        verbose: bool = False
    ):
        """
        Initialize PSG preprocessor.
        
        Parameters:
        -----------
        target_sfreq : float
            Target sampling frequency (default: 100 Hz)
        notch_freq : float or None
            Notch filter frequency for line noise removal (50Hz or 60Hz)
            Set to None to disable notch filtering
        notch_quality : float
            Quality factor for notch filter (higher = narrower)
        filter_method : str
            Filter method ('fir' or 'iir')
        filter_length : str
            Filter length ('auto', '10s', etc.)
        verbose : bool
            Whether to print verbose output
        """
        self.target_sfreq = target_sfreq
        self.notch_freq = notch_freq
        self.notch_quality = notch_quality
        self.filter_method = filter_method
        self.filter_length = filter_length
        self.verbose = verbose
        
        # Default frequency bands for different signal types
        self.freq_bands = {
            'eeg': (0.5, 35.0),      # EEG: 0.5-35 Hz
            'eog': (0.1, 15.0),      # EOG: 0.1-15 Hz
            'emg': (10.0, 100.0),    # EMG: 10-100 Hz
            'ecg': (0.5, 40.0),      # ECG: 0.5-40 Hz
            'resp': (0.1, 5.0),      # Respiratory: 0.1-5 Hz
            'temp': (0.01, 0.5),     # Temperature: 0.01-0.5 Hz
            'default': (0.5, 35.0)   # Default: 0.5-35 Hz
        }
        
    def detect_signal_type(self, ch_name: str) -> str:
        """
        Detect signal type from channel name.
        
        Parameters:
        -----------
        ch_name : str
            Channel name
        
        Returns:
        --------
        str : Signal type ('eeg', 'eog', 'emg', 'ecg', 'resp', 'temp', 'default')
        """
        ch_name_lower = ch_name.lower()
        
        if 'eeg' in ch_name_lower:
            return 'eeg'
        elif 'eog' in ch_name_lower:
            return 'eog'
        elif 'emg' in ch_name_lower:
            return 'emg'
        elif 'ecg' in ch_name_lower or 'ekg' in ch_name_lower:
            return 'ecg'
        elif 'resp' in ch_name_lower or 'breath' in ch_name_lower:
            return 'resp'
        elif 'temp' in ch_name_lower or 'temperature' in ch_name_lower:
            return 'temp'
        else:
            return 'default'
    
    def resample_raw(self, raw: mne.io.Raw, target_sfreq: Optional[float] = None) -> mne.io.Raw:
        """
        Resample raw data to target sampling frequency.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        target_sfreq : float, optional
            Target sampling frequency (uses self.target_sfreq if None)
        
        Returns:
        --------
        mne.io.Raw : Resampled raw object
        """
        if target_sfreq is None:
            target_sfreq = self.target_sfreq
        
        original_sfreq = raw.info['sfreq']
        
        if abs(original_sfreq - target_sfreq) < 0.1:
            if self.verbose:
                print(f"Sampling frequency already at {target_sfreq} Hz, skipping resampling")
            return raw
        
        if self.verbose:
            print(f"Resampling from {original_sfreq:.2f} Hz to {target_sfreq:.2f} Hz")
        
        raw_resampled = raw.copy()
        raw_resampled.resample(target_sfreq, npad='auto', verbose=self.verbose)
        
        return raw_resampled
    
    def apply_notch_filter(
        self,
        raw: mne.io.Raw,
        freqs: Optional[Union[float, List[float]]] = None,
        notch_widths: Optional[Union[float, List[float]]] = None
    ) -> mne.io.Raw:
        """
        Apply notch filter to remove line noise (50/60 Hz).
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        freqs : float or list of float, optional
            Notch frequencies (uses self.notch_freq if None)
        notch_widths : float or list of float, optional
            Notch widths (automatically calculated if None)
        
        Returns:
        --------
        mne.io.Raw : Filtered raw object
        """
        if freqs is None:
            freqs = self.notch_freq
        
        if freqs is None:
            if self.verbose:
                print("Notch filtering disabled")
            return raw
        
        # Convert single frequency to list
        if isinstance(freqs, (int, float)):
            freqs = [freqs]
        
        # Calculate notch widths if not provided
        if notch_widths is None:
            notch_widths = [freq / self.notch_quality for freq in freqs]
        elif isinstance(notch_widths, (int, float)):
            notch_widths = [notch_widths]
        
        if self.verbose:
            print(f"Applying notch filter at {freqs} Hz")
        
        raw_filtered = raw.copy()
        raw_filtered.notch_filter(
            freqs=freqs,
            notch_widths=notch_widths,
            filter_length=self.filter_length,
            method=self.filter_method,
            verbose=self.verbose
        )
        
        return raw_filtered
    
    def apply_bandpass_filter(
        self,
        raw: mne.io.Raw,
        channel_specific: bool = True,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None
    ) -> mne.io.Raw:
        """
        Apply bandpass filter to raw data.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        channel_specific : bool
            Whether to apply channel-specific frequency bands
        low_freq : float, optional
            Low cutoff frequency (uses signal-specific default if None)
        high_freq : float, optional
            High cutoff frequency (uses signal-specific default if None)
        
        Returns:
        --------
        mne.io.Raw : Filtered raw object
        """
        raw_filtered = raw.copy()
        
        if channel_specific:
            # Apply channel-specific filtering
            if self.verbose:
                print("Applying channel-specific bandpass filters")
            
            # Group channels by type
            channels_by_type = {}
            for ch_idx, ch_name in enumerate(raw.ch_names):
                ch_type = self.detect_signal_type(ch_name)
                if ch_type not in channels_by_type:
                    channels_by_type[ch_type] = []
                channels_by_type[ch_type].append(ch_idx)
            
            # Apply filters to each group
            for ch_type, ch_indices in channels_by_type.items():
                low, high = self.freq_bands.get(ch_type, self.freq_bands['default'])
                
                if self.verbose:
                    ch_names = [raw.ch_names[i] for i in ch_indices]
                    print(f"  Filtering {ch_type} channels ({', '.join(ch_names)}): "
                          f"{low}-{high} Hz")
                
                # Pick channels and filter
                picks = mne.pick_channels(raw.ch_names, include=[raw.ch_names[i] for i in ch_indices])
                raw_filtered.filter(
                    l_freq=low,
                    h_freq=high,
                    picks=picks,
                    filter_length=self.filter_length,
                    method=self.filter_method,
                    verbose=self.verbose
                )
        else:
            # Apply same filter to all channels
            if low_freq is None:
                low_freq = self.freq_bands['default'][0]
            if high_freq is None:
                high_freq = self.freq_bands['default'][1]
            
            if self.verbose:
                print(f"Applying bandpass filter ({low_freq}-{high_freq} Hz) to all channels")
            
            raw_filtered.filter(
                l_freq=low_freq,
                h_freq=high_freq,
                filter_length=self.filter_length,
                method=self.filter_method,
                verbose=self.verbose
            )
        
        return raw_filtered
    
    def detect_bad_channels(
        self,
        raw: mne.io.Raw,
        method: str = 'auto',
        flat_criteria: Dict = None,
        noisy_criteria: Dict = None
    ) -> List[str]:
        """
        Detect bad channels using various criteria.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        method : str
            Detection method ('auto', 'flat', 'noisy', 'correlation', 'all')
        flat_criteria : dict
            Criteria for flat channel detection
        noisy_criteria : dict
            Criteria for noisy channel detection
        
        Returns:
        --------
        list : List of bad channel names
        """
        if flat_criteria is None:
            flat_criteria = {'min_std': 1e-6}
        if noisy_criteria is None:
            noisy_criteria = {'bad_percent': 5}
        
        bad_channels = []
        
        # Get existing bad channels
        if raw.info['bads']:
            bad_channels.extend(raw.info['bads'])
        
        if method in ['auto', 'flat', 'all']:
            # Detect flat channels (very low variance)
            data = raw.get_data()
            channel_stds = np.std(data, axis=1)
            flat_threshold = flat_criteria.get('min_std', 1e-6)
            
            flat_channels = [
                raw.ch_names[i] for i in range(len(raw.ch_names))
                if channel_stds[i] < flat_threshold
            ]
            bad_channels.extend(flat_channels)
            
            if self.verbose and flat_channels:
                print(f"Detected flat channels: {flat_channels}")
        
        if method in ['auto', 'noisy', 'all']:
            # Detect noisy channels (high variance or artifacts)
            data = raw.get_data()
            channel_stds = np.std(data, axis=1)
            channel_means = np.mean(np.abs(data), axis=1)
            
            # Channels with very high variance relative to mean
            noisy_threshold = np.percentile(channel_stds, 100 - noisy_criteria.get('bad_percent', 5))
            noisy_channels = [
                raw.ch_names[i] for i in range(len(raw.ch_names))
                if channel_stds[i] > noisy_threshold * 3  # 3x the threshold
            ]
            bad_channels.extend(noisy_channels)
            
            if self.verbose and noisy_channels:
                print(f"Detected noisy channels: {noisy_channels}")
        
        if method in ['auto', 'correlation', 'all']:
            # Detect channels with low correlation to other channels
            data = raw.get_data()
            # Downsample for faster correlation computation
            if data.shape[1] > 10000:
                step = data.shape[1] // 10000
                data = data[:, ::step]
            
            correlations = []
            for i in range(len(raw.ch_names)):
                if raw.ch_names[i] in bad_channels:
                    correlations.append(0)
                    continue
                
                # Compute correlation with other channels
                other_indices = [j for j in range(len(raw.ch_names)) 
                               if j != i and raw.ch_names[j] not in bad_channels]
                if len(other_indices) > 0:
                    corr = np.corrcoef(data[i], data[other_indices].mean(axis=0))[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            
            # Channels with very low correlation
            low_corr_threshold = np.percentile(correlations, 10)
            low_corr_channels = [
                raw.ch_names[i] for i in range(len(raw.ch_names))
                if correlations[i] < low_corr_threshold * 0.5 and correlations[i] < 0.3
            ]
            bad_channels.extend(low_corr_channels)
            
            if self.verbose and low_corr_channels:
                print(f"Detected low-correlation channels: {low_corr_channels}")
        
        # Remove duplicates
        bad_channels = list(set(bad_channels))
        
        if self.verbose:
            print(f"Total bad channels detected: {bad_channels}")
        
        return bad_channels
    
    def interpolate_bad_channels(
        self,
        raw: mne.io.Raw,
        bad_channels: Optional[List[str]] = None,
        method: str = 'MNE'
    ) -> mne.io.Raw:
        """
        Interpolate bad channels using surrounding good channels.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        bad_channels : list of str, optional
            List of bad channel names (uses raw.info['bads'] if None)
        method : str
            Interpolation method ('MNE' or 'scipy')
        
        Returns:
        --------
        mne.io.Raw : Raw object with interpolated channels
        """
        if bad_channels is None:
            bad_channels = raw.info['bads']
        
        if not bad_channels:
            if self.verbose:
                print("No bad channels to interpolate")
            return raw
        
        raw_interp = raw.copy()
        
        if method == 'MNE':
            # Use MNE's interpolation (recommended for EEG/EOG)
            if self.verbose:
                print(f"Interpolating bad channels using MNE: {bad_channels}")
            
            try:
                raw_interp.interpolate_bads(reset_bads=True, verbose=self.verbose)
            except Exception as e:
                if self.verbose:
                    print(f"MNE interpolation failed: {e}, trying alternative method")
                # Fall back to scipy method
                method = 'scipy'
        
        if method == 'scipy':
            # Use scipy interpolation (fallback)
            if self.verbose:
                print(f"Interpolating bad channels using scipy: {bad_channels}")
            
            data = raw_interp.get_data()
            good_channels = [ch for ch in raw_interp.ch_names if ch not in bad_channels]
            
            if len(good_channels) < 2:
                if self.verbose:
                    print("Not enough good channels for interpolation, skipping")
                return raw_interp
            
            # Get good channel data
            good_indices = [raw_interp.ch_names.index(ch) for ch in good_channels]
            good_data = data[good_indices]
            
            # Interpolate each bad channel as average of good channels
            for bad_ch in bad_channels:
                bad_idx = raw_interp.ch_names.index(bad_ch)
                # Simple interpolation: average of good channels
                data[bad_idx] = np.mean(good_data, axis=0)
            
            # Update raw object
            raw_interp._data = data
            raw_interp.info['bads'] = []
        
        return raw_interp
    
    def remove_artifacts_ica(
        self,
        raw: mne.io.Raw,
        n_components: Optional[int] = None,
        method: str = 'fastica',
        ecg_ch_name: Optional[str] = None,
        eog_ch_name: Optional[str] = None,
        reject: Optional[Dict] = None
    ) -> mne.io.Raw:
        """
        Remove artifacts using Independent Component Analysis (ICA).
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        n_components : int, optional
            Number of ICA components (uses min(n_channels, 20) if None)
        method : str
            ICA method ('fastica', 'infomax', 'picard')
        ecg_ch_name : str, optional
            ECG channel name for ECG artifact detection
        eog_ch_name : str, optional
            EOG channel name for EOG artifact detection
        reject : dict, optional
            Rejection parameters for epoching
        
        Returns:
        --------
        mne.io.Raw : Raw object with artifacts removed
        """
        if self.verbose:
            print("Applying ICA for artifact removal")
        
        # Pick only EEG/EOG channels for ICA (skip EMG, ECG, etc.)
        picks = mne.pick_types(
            raw.info,
            meg=False,
            eeg=True,
            eog=True,
            ecg=False,
            emg=False,
            exclude='bads'
        )
        
        if len(picks) < 2:
            if self.verbose:
                print("Not enough channels for ICA, skipping")
            return raw
        
        if n_components is None:
            n_components = min(len(picks), 20)
        
        # Create and fit ICA
        ica = ICA(
            n_components=n_components,
            method=method,
            random_state=42,
            verbose=self.verbose
        )
        
        # Fit ICA (use a subset of data if very long)
        if raw.times[-1] > 3600:  # More than 1 hour
            raw_short = raw.copy().crop(tmax=3600)
            ica.fit(raw_short, picks=picks, verbose=self.verbose)
        else:
            ica.fit(raw, picks=picks, verbose=self.verbose)
        
        # Detect artifacts
        artifact_components = []
        
        # Detect ECG artifacts
        if ecg_ch_name and ecg_ch_name in raw.ch_names:
            try:
                ecg_epochs = create_ecg_epochs(raw, ch_name=ecg_ch_name, reject=reject)
                ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name=ecg_ch_name)
                artifact_components.extend(ecg_inds)
                if self.verbose:
                    print(f"  Detected {len(ecg_inds)} ECG artifact components")
            except Exception as e:
                if self.verbose:
                    print(f"  ECG artifact detection failed: {e}")
        
        # Detect EOG artifacts
        if eog_ch_name and eog_ch_name in raw.ch_names:
            try:
                eog_epochs = create_eog_epochs(raw, ch_name=eog_ch_name, reject=reject)
                eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=eog_ch_name)
                artifact_components.extend(eog_inds)
                if self.verbose:
                    print(f"  Detected {len(eog_inds)} EOG artifact components")
            except Exception as e:
                if self.verbose:
                    print(f"  EOG artifact detection failed: {e}")
        
        # Remove duplicate components
        artifact_components = list(set(artifact_components))
        
        if artifact_components:
            if self.verbose:
                print(f"Removing {len(artifact_components)} artifact components: {artifact_components}")
            ica.exclude = artifact_components
            raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components, verbose=self.verbose)
        else:
            if self.verbose:
                print("No artifact components detected")
            raw_cleaned = raw
        
        return raw_cleaned
    
    def remove_artifacts_threshold(
        self,
        raw: mne.io.Raw,
        threshold: float = 5.0,
        method: str = 'zscore'
    ) -> mne.io.Raw:
        """
        Remove artifacts using threshold-based method.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        threshold : float
            Threshold for artifact detection (in standard deviations)
        method : str
            Method ('zscore', 'mad' for median absolute deviation)
        
        Returns:
        --------
        mne.io.Raw : Raw object with artifacts removed (clipped or set to NaN)
        """
        if self.verbose:
            print(f"Applying threshold-based artifact removal (threshold={threshold}, method={method})")
        
        raw_cleaned = raw.copy()
        data = raw_cleaned.get_data()
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            ch_data = data[ch_idx]
            
            if method == 'zscore':
                mean = np.mean(ch_data)
                std = np.std(ch_data)
                z_scores = np.abs((ch_data - mean) / std)
                artifacts = z_scores > threshold
            elif method == 'mad':
                median = np.median(ch_data)
                mad = np.median(np.abs(ch_data - median))
                z_scores = np.abs((ch_data - median) / (mad * 1.4826))  # MAD to SD conversion
                artifacts = z_scores > threshold
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if np.any(artifacts):
                # Clip artifacts to threshold
                if method == 'zscore':
                    data[ch_idx, artifacts] = np.sign(ch_data[artifacts]) * threshold * std + mean
                else:
                    data[ch_idx, artifacts] = np.sign(ch_data[artifacts]) * threshold * mad * 1.4826 + median
                
                if self.verbose:
                    artifact_percent = 100 * np.sum(artifacts) / len(artifacts)
                    print(f"  {ch_name}: {artifact_percent:.2f}% artifacts removed")
        
        raw_cleaned._data = data
        return raw_cleaned
    
    def preprocess(
        self,
        raw: mne.io.Raw,
        steps: List[str] = None,
        resample: bool = True,
        notch_filter: bool = True,
        bandpass_filter: bool = True,
        channel_specific_filter: bool = True,
        detect_bads: bool = True,
        interpolate_bads: bool = True,
        remove_artifacts: bool = True,
        artifact_method: str = 'threshold',
        **kwargs
    ) -> mne.io.Raw:
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Raw MNE object
        steps : list of str, optional
            List of preprocessing steps to apply (uses all if None)
        resample : bool
            Whether to resample to target frequency
        notch_filter : bool
            Whether to apply notch filter
        bandpass_filter : bool
            Whether to apply bandpass filter
        channel_specific_filter : bool
            Whether to use channel-specific frequency bands
        detect_bads : bool
            Whether to detect bad channels
        interpolate_bads : bool
            Whether to interpolate bad channels
        remove_artifacts : bool
            Whether to remove artifacts
        artifact_method : str
            Artifact removal method ('threshold', 'ica', or 'both')
        **kwargs : dict
            Additional parameters for specific steps
        
        Returns:
        --------
        mne.io.Raw : Preprocessed raw object
        """
        if steps is None:
            steps = ['resample', 'notch', 'bandpass', 'detect_bads', 'interpolate', 'artifacts']
        
        processed = raw.copy()
        
        if self.verbose:
            print("=" * 60)
            print("Starting PSG Preprocessing Pipeline")
            print("=" * 60)
            print(f"Original sampling frequency: {processed.info['sfreq']:.2f} Hz")
            print(f"Number of channels: {len(processed.ch_names)}")
            print(f"Duration: {processed.times[-1]:.2f} seconds")
            print("=" * 60)
        
        # Step 1: Resample
        if 'resample' in steps and resample:
            if self.verbose:
                print("\n[Step 1/6] Resampling...")
            processed = self.resample_raw(processed)
        
        # Step 2: Notch filter
        if 'notch' in steps and notch_filter:
            if self.verbose:
                print("\n[Step 2/6] Applying notch filter...")
            processed = self.apply_notch_filter(processed)
        
        # Step 3: Bandpass filter
        if 'bandpass' in steps and bandpass_filter:
            if self.verbose:
                print("\n[Step 3/6] Applying bandpass filter...")
            processed = self.apply_bandpass_filter(
                processed,
                channel_specific=channel_specific_filter,
                low_freq=kwargs.get('low_freq'),
                high_freq=kwargs.get('high_freq')
            )
        
        # Step 4: Detect bad channels
        if 'detect_bads' in steps and detect_bads:
            if self.verbose:
                print("\n[Step 4/6] Detecting bad channels...")
            bad_channels = self.detect_bad_channels(
                processed,
                method=kwargs.get('bad_detection_method', 'auto'),
                flat_criteria=kwargs.get('flat_criteria'),
                noisy_criteria=kwargs.get('noisy_criteria')
            )
            processed.info['bads'] = bad_channels
        
        # Step 5: Interpolate bad channels
        if 'interpolate' in steps and interpolate_bads:
            if self.verbose:
                print("\n[Step 5/6] Interpolating bad channels...")
            processed = self.interpolate_bad_channels(
                processed,
                method=kwargs.get('interpolation_method', 'MNE')
            )
        
        # Step 6: Remove artifacts
        if 'artifacts' in steps and remove_artifacts:
            if self.verbose:
                print("\n[Step 6/6] Removing artifacts...")
            
            if artifact_method == 'threshold':
                processed = self.remove_artifacts_threshold(
                    processed,
                    threshold=kwargs.get('artifact_threshold', 5.0),
                    method=kwargs.get('threshold_method', 'zscore')
                )
            elif artifact_method == 'ica':
                processed = self.remove_artifacts_ica(
                    processed,
                    n_components=kwargs.get('ica_components'),
                    method=kwargs.get('ica_method', 'fastica'),
                    ecg_ch_name=kwargs.get('ecg_ch_name'),
                    eog_ch_name=kwargs.get('eog_ch_name'),
                    reject=kwargs.get('reject')
                )
            elif artifact_method == 'both':
                # Apply threshold first, then ICA
                processed = self.remove_artifacts_threshold(
                    processed,
                    threshold=kwargs.get('artifact_threshold', 5.0),
                    method=kwargs.get('threshold_method', 'zscore')
                )
                processed = self.remove_artifacts_ica(
                    processed,
                    n_components=kwargs.get('ica_components'),
                    method=kwargs.get('ica_method', 'fastica'),
                    ecg_ch_name=kwargs.get('ecg_ch_name'),
                    eog_ch_name=kwargs.get('eog_ch_name'),
                    reject=kwargs.get('reject')
                )
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("Preprocessing Complete!")
            print("=" * 60)
            print(f"Final sampling frequency: {processed.info['sfreq']:.2f} Hz")
            print(f"Number of channels: {len(processed.ch_names)}")
            print(f"Bad channels: {processed.info['bads']}")
            print(f"Duration: {processed.times[-1]:.2f} seconds")
            print("=" * 60)
        
        return processed


def preprocess_psg_file(
    file_path: str,
    output_path: Optional[str] = None,
    target_sfreq: float = 100.0,
    notch_freq: Optional[float] = 50.0,
    channel_specific_filter: bool = True,
    artifact_method: str = 'threshold',
    save_preprocessed: bool = True,
    verbose: bool = True,
    **kwargs
) -> mne.io.Raw:
    """
    Preprocess a single PSG file.
    
    Parameters:
    -----------
    file_path : str
        Path to PSG file (EDF format)
    output_path : str, optional
        Path to save preprocessed file (uses input path with '_preprocessed' suffix if None)
    target_sfreq : float
        Target sampling frequency
    notch_freq : float or None
        Notch filter frequency
    channel_specific_filter : bool
        Whether to use channel-specific frequency bands
    artifact_method : str
        Artifact removal method
    save_preprocessed : bool
        Whether to save preprocessed file
    verbose : bool
        Whether to print verbose output
    **kwargs : dict
        Additional parameters for preprocessing
    
    Returns:
    --------
    mne.io.Raw : Preprocessed raw object
    """
    if verbose:
        print(f"Loading file: {file_path}")
    
    # Load raw data
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Create preprocessor
    preprocessor = PSGPreprocessor(
        target_sfreq=target_sfreq,
        notch_freq=notch_freq,
        verbose=verbose
    )
    
    # Preprocess
    processed = preprocessor.preprocess(
        raw,
        resample=True,
        notch_filter=True,
        bandpass_filter=True,
        channel_specific_filter=channel_specific_filter,
        detect_bads=True,
        interpolate_bads=True,
        remove_artifacts=True,
        artifact_method=artifact_method,
        **kwargs
    )
    
    # Save if requested
    if save_preprocessed:
        if output_path is None:
            base, ext = os.path.splitext(file_path)
            output_path = f"{base}_preprocessed{ext}"
        
        if verbose:
            print(f"Saving preprocessed file: {output_path}")
        
        processed.save(output_path, overwrite=True, verbose=False)
    
    return processed


def preprocess_psg_folder(
    folder_path: str,
    output_folder: Optional[str] = None,
    file_pattern: str = "*-PSG.edf",
    target_sfreq: float = 100.0,
    notch_freq: Optional[float] = 50.0,
    channel_specific_filter: bool = True,
    artifact_method: str = 'threshold',
    verbose: bool = True,
    **kwargs
) -> List[mne.io.Raw]:
    """
    Preprocess all PSG files in a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing PSG files
    output_folder : str, optional
        Path to save preprocessed files (uses input folder if None)
    file_pattern : str
        File pattern to match (e.g., "*-PSG.edf")
    target_sfreq : float
        Target sampling frequency
    notch_freq : float or None
        Notch filter frequency
    channel_specific_filter : bool
        Whether to use channel-specific frequency bands
    artifact_method : str
        Artifact removal method
    verbose : bool
        Whether to print verbose output
    **kwargs : dict
        Additional parameters for preprocessing
    
    Returns:
    --------
    list : List of preprocessed raw objects
    """
    import glob
    
    if output_folder is None:
        output_folder = folder_path
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # Find all PSG files
    pattern = os.path.join(folder_path, file_pattern)
    psg_files = glob.glob(pattern)
    
    if verbose:
        print(f"Found {len(psg_files)} PSG files in {folder_path}")
    
    preprocessed_files = []
    
    for file_path in psg_files:
        try:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_folder, filename.replace('.edf', '_preprocessed.edf'))
            
            processed = preprocess_psg_file(
                file_path=file_path,
                output_path=output_path,
                target_sfreq=target_sfreq,
                notch_freq=notch_freq,
                channel_specific_filter=channel_specific_filter,
                artifact_method=artifact_method,
                save_preprocessed=True,
                verbose=verbose,
                **kwargs
            )
            
            preprocessed_files.append(processed)
            
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            continue
    
    if verbose:
        print(f"\nSuccessfully preprocessed {len(preprocessed_files)} files")
    
    return preprocessed_files


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess PSG files")
    parser.add_argument("--input", type=str, required=True, help="Input file or folder")
    parser.add_argument("--output", type=str, default=None, help="Output file or folder")
    parser.add_argument("--target_sfreq", type=float, default=100.0, help="Target sampling frequency")
    parser.add_argument("--notch_freq", type=float, default=50.0, help="Notch filter frequency (50 or 60 Hz)")
    parser.add_argument("--artifact_method", type=str, default="threshold", 
                       choices=["threshold", "ica", "both"], help="Artifact removal method")
    parser.add_argument("--channel_specific", action="store_true", 
                       help="Use channel-specific frequency bands")
    parser.add_argument("--folder", action="store_true", help="Process entire folder")
    parser.add_argument("--pattern", type=str, default="*-PSG.edf", help="File pattern for folder processing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.folder:
        preprocess_psg_folder(
            folder_path=args.input,
            output_folder=args.output,
            file_pattern=args.pattern,
            target_sfreq=args.target_sfreq,
            notch_freq=args.notch_freq,
            channel_specific_filter=args.channel_specific,
            artifact_method=args.artifact_method,
            verbose=args.verbose
        )
    else:
        preprocess_psg_file(
            file_path=args.input,
            output_path=args.output,
            target_sfreq=args.target_sfreq,
            notch_freq=args.notch_freq,
            channel_specific_filter=args.channel_specific,
            artifact_method=args.artifact_method,
            verbose=args.verbose
        )

