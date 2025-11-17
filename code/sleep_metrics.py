"""
Sleep Metrics Calculator from EDF Files
Calculates: AHI, ODI, Min SpO2, Mean SpO2, RDI, Sleep Latency, 
WASO, Sleep Efficiency, Arousal Index
Reads sleep stages from XML annotation files
"""

import numpy as np
import mne
from datetime import datetime, timedelta
import pandas as pd
from scipy import signal
from typing import Dict, Tuple, List
import xml.etree.ElementTree as ET

class SleepMetricsCalculator:
    def __init__(self, edf_path: str, xml_path: str = None, hypnogram: np.ndarray = None, epoch_length: int = 30):
        """
        Initialize the calculator with EDF file and sleep stages.
        
        Parameters:
        -----------
        edf_path : str
            Path to the EDF file
        xml_path : str, optional
            Path to XML file containing sleep stage annotations
        hypnogram : np.ndarray, optional
            Sleep stages array (alternative to XML file)
            Format: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, -1=Unknown
        epoch_length : int
            Length of each epoch in seconds (default: 30)
        """
        self.edf_path = edf_path
        self.epoch_length = epoch_length
        self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        self.sfreq = self.raw.info['sfreq']
        
        # Load hypnogram from XML or use provided array
        if xml_path is not None:
            self.hypnogram = self.load_hypnogram_from_xml(xml_path)
        elif hypnogram is not None:
            self.hypnogram = hypnogram
        else:
            raise ValueError("Either xml_path or hypnogram must be provided")
    
    def load_hypnogram_from_xml(self, xml_path: str) -> np.ndarray:
        """
        Load sleep stage annotations from XML file.
        Supports multiple XML formats (NSRR, EDF+, PSG annotation formats).
        
        Returns:
        --------
        hypnogram : np.ndarray
            Array of sleep stages (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, -1=Unknown)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Detect XML format and parse accordingly
            if root.tag == 'PSGAnnotation' or 'PSG' in root.tag:
                return self._parse_nsrr_xml(root)
            elif root.tag == 'CMPStudyConfig' or 'Compumedics' in root.tag:
                return self._parse_compumedics_xml(root)
            elif 'ScoredEvents' in root.tag or self._find_element(root, 'ScoredEvent') is not None:
                return self._parse_scored_events_xml(root)
            else:
                # Try generic parsing
                return self._parse_generic_xml(root)
                
        except Exception as e:
            raise ValueError(f"Error parsing XML file: {str(e)}\n"
                           f"Please ensure the XML file is a valid sleep stage annotation file.")
    
    def _find_element(self, root, tag):
        """Recursively find an element in XML tree."""
        for elem in root.iter():
            if tag in elem.tag:
                return elem
        return None
    
    def _parse_nsrr_xml(self, root) -> np.ndarray:
        """Parse NSRR (National Sleep Research Resource) format XML."""
        epochs = []
        
        # Find all ScoredEvents - try multiple possible paths
        scored_events = (root.findall('.//ScoredEvent') or 
                        root.findall('.//ScoredEvents/ScoredEvent') or
                        root.findall('ScoredEvent'))
        
        print(f"Found {len(scored_events)} scored events in XML")
        
        for event in scored_events:
            event_type = event.find('EventType')
            event_concept = event.find('EventConcept')
            start = event.find('Start')
            duration = event.find('Duration')
            
            # Check if this is a sleep stage event
            if event_type is not None and 'stage' in (event_type.text or '').lower():
                if event_concept is not None and start is not None and duration is not None:
                    # Parse the EventConcept which contains "Stage Name|Number"
                    stage_text = event_concept.text or ''
                    start_time = float(start.text)
                    dur = float(duration.text)
                    
                    # Debug: print first few events
                    if len(epochs) < 5:
                        print(f"  Event: '{stage_text}' at {start_time}s for {dur}s")
                    
                    stage = self._parse_stage_label(stage_text)
                    epochs.append({
                        'start': start_time,
                        'duration': dur,
                        'stage': stage
                    })
        
        if not epochs:
            raise ValueError("No sleep stage events found in XML")
        
        print(f"Parsed {len(epochs)} sleep stage events")
        return self._epochs_to_hypnogram(epochs)
    
    def _parse_compumedics_xml(self, root) -> np.ndarray:
        """Parse Compumedics Profusion XML format."""
        epochs = []
        
        for event in root.findall('.//SleepStage'):
            start = event.get('Start') or event.find('Start')
            duration = event.get('Duration') or event.find('Duration')
            stage_text = event.get('Stage') or event.text
            
            if start is not None and duration is not None and stage_text is not None:
                start_time = float(start.text if hasattr(start, 'text') else start)
                dur = float(duration.text if hasattr(duration, 'text') else duration)
                stage = self._parse_stage_label(stage_text)
                
                epochs.append({
                    'start': start_time,
                    'duration': dur,
                    'stage': stage
                })
        
        if not epochs:
            raise ValueError("No sleep stage events found in XML")
        
        return self._epochs_to_hypnogram(epochs)
    
    def _parse_scored_events_xml(self, root) -> np.ndarray:
        """Parse generic ScoredEvents XML format."""
        epochs = []
        
        # Try multiple possible paths
        event_paths = [
            './/ScoredEvent',
            './/Event',
            './/SleepStage',
            './/Epoch'
        ]
        
        for path in event_paths:
            events = root.findall(path)
            if events:
                break
        
        for event in events:
            # Try to find stage information
            stage_elem = (event.find('EventType') or 
                         event.find('EventConcept') or
                         event.find('Stage') or
                         event.find('Type'))
            
            start_elem = event.find('Start') or event.find('StartTime')
            duration_elem = event.find('Duration') or event.find('Length')
            
            if stage_elem is not None and start_elem is not None:
                stage_text = stage_elem.text or stage_elem.get('value', '')
                start_time = float(start_elem.text or start_elem.get('value', 0))
                
                # Duration might not always be present, default to epoch_length
                if duration_elem is not None:
                    dur = float(duration_elem.text or duration_elem.get('value', self.epoch_length))
                else:
                    dur = self.epoch_length
                
                stage = self._parse_stage_label(stage_text)
                
                epochs.append({
                    'start': start_time,
                    'duration': dur,
                    'stage': stage
                })
        
        if not epochs:
            raise ValueError("No sleep stage events found in XML")
        
        return self._epochs_to_hypnogram(epochs)
    
    def _parse_generic_xml(self, root) -> np.ndarray:
        """Parse generic XML format by searching for common patterns."""
        epochs = []
        
        # Search for any element containing stage/sleep information
        for elem in root.iter():
            text = (elem.text or '').strip().lower()
            tag = elem.tag.lower()
            
            # Check if this element contains stage information
            if any(stage in text for stage in ['wake', 'rem', 'n1', 'n2', 'n3', 'stage']):
                # Try to find associated time information
                start = None
                duration = None
                
                # Check attributes
                if 'start' in elem.attrib:
                    start = float(elem.attrib['start'])
                if 'duration' in elem.attrib:
                    duration = float(elem.attrib['duration'])
                
                # Check child elements
                if start is None:
                    for child in elem:
                        if 'start' in child.tag.lower() and child.text:
                            start = float(child.text)
                        if 'duration' in child.tag.lower() and child.text:
                            duration = float(child.text)
                
                if start is not None:
                    stage = self._parse_stage_label(text)
                    epochs.append({
                        'start': start,
                        'duration': duration or self.epoch_length,
                        'stage': stage
                    })
        
        if not epochs:
            raise ValueError("Could not parse XML file. Unknown format.")
        
        return self._epochs_to_hypnogram(epochs)
    
    def _parse_stage_label(self, label: str) -> int:
        """
        Convert stage label string to numeric code.
        
        Returns:
        --------
        stage : int
            0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, -1=Unknown
        """
        label = label.lower().strip()
        
        # Wake
        if 'wake' in label or label in ['w', '0']:
            return 0
        # N1 / Stage 1
        elif 'n1' in label or 'stage 1' in label or 'nrem1' in label or label == '1':
            return 1
        # N2 / Stage 2
        elif 'n2' in label or 'stage 2' in label or 'nrem2' in label or label == '2':
            return 2
        # N3 / Stage 3 / Stage 4 (or N4, SWS)
        elif ('n3' in label or 'n4' in label or 'stage 3' in label or 
              'stage 4' in label or 'nrem3' in label or 'nrem4' in label or 
              'sws' in label or label in ['3', '4']):
            return 3
        # REM
        elif 'rem' in label or label == 'r' or label == '5':
            return 4
        # Unknown
        else:
            print(f"Warning: Unknown stage label: '{label}'")
            return -1
    
    def _epochs_to_hypnogram(self, epochs: List[Dict]) -> np.ndarray:
        """
        Convert list of epoch dictionaries to hypnogram array.
        
        Parameters:
        -----------
        epochs : list of dict
            Each dict contains 'start', 'duration', and 'stage'
        
        Returns:
        --------
        hypnogram : np.ndarray
            Array of sleep stages
        """
        # Sort epochs by start time
        epochs = sorted(epochs, key=lambda x: x['start'])
        
        # Determine total recording duration
        last_epoch = epochs[-1]
        total_duration = last_epoch['start'] + last_epoch['duration']
        
        # Create hypnogram array
        num_epochs = int(np.ceil(total_duration / self.epoch_length))
        hypnogram = np.full(num_epochs, -1, dtype=int)  # Initialize with unknown
        
        # Fill in sleep stages
        for epoch in epochs:
            start_idx = int(epoch['start'] / self.epoch_length)
            num_epoch_segments = int(np.ceil(epoch['duration'] / self.epoch_length))
            
            for i in range(num_epoch_segments):
                idx = start_idx + i
                if idx < len(hypnogram):
                    hypnogram[idx] = epoch['stage']
        
        return hypnogram
        
    def get_channel_data(self, channel_patterns: List[str]) -> Tuple[np.ndarray, str]:
        """Get data from the first matching channel."""
        ch_names = self.raw.ch_names
        for pattern in channel_patterns:
            matches = [ch for ch in ch_names if pattern.lower() in ch.lower()]
            if matches:
                data, _ = self.raw[matches[0], :]
                return data[0], matches[0]
        return None, None
    
    def detect_apneas_hypopneas(self, resp_signal: np.ndarray, 
                                spo2_signal: np.ndarray = None) -> Dict:
        """
        Detect apneas and hypopneas from respiratory signal.
        Simplified detection based on amplitude reduction.
        """
        # Downsample to 1 Hz for easier processing
        target_freq = 1
        if self.sfreq != target_freq:
            resp_resampled = signal.resample(
                resp_signal, 
                int(len(resp_signal) * target_freq / self.sfreq)
            )
        else:
            resp_resampled = resp_signal
        
        # Calculate baseline using moving window
        window_size = 120  # 2 minutes
        baseline = pd.Series(np.abs(resp_resampled)).rolling(
            window=window_size, center=True, min_periods=1
        ).median().values
        
        # Detect events (>50% reduction for >10 seconds)
        amplitude_ratio = np.abs(resp_resampled) / (baseline + 1e-10)
        min_duration = 10  # seconds
        
        apneas = []  # >90% reduction
        hypopneas = []  # 50-90% reduction
        
        in_event = False
        event_start = 0
        event_type = None
        
        for i in range(len(amplitude_ratio)):
            if amplitude_ratio[i] < 0.1 and not in_event:  # Apnea start
                in_event = True
                event_start = i
                event_type = 'apnea'
            elif 0.1 <= amplitude_ratio[i] < 0.5 and not in_event:  # Hypopnea start
                in_event = True
                event_start = i
                event_type = 'hypopnea'
            elif amplitude_ratio[i] >= 0.5 and in_event:  # Event end
                duration = i - event_start
                if duration >= min_duration:
                    if event_type == 'apnea':
                        apneas.append((event_start, i))
                    else:
                        hypopneas.append((event_start, i))
                in_event = False
                event_type = None
        
        return {
            'apneas': apneas,
            'hypopneas': hypopneas,
            'total_events': len(apneas) + len(hypopneas)
        }
    
    def detect_desaturations(self, spo2_signal: np.ndarray, threshold: float = 3.0) -> List:
        """
        Detect oxygen desaturations (ODI).
        
        Parameters:
        -----------
        spo2_signal : np.ndarray
            SpO2 signal
        threshold : float
            Desaturation threshold in % (default: 3%)
        """
        # Downsample to 1 Hz
        if self.sfreq != 1:
            spo2_resampled = signal.resample(
                spo2_signal, 
                int(len(spo2_signal) / self.sfreq)
            )
        else:
            spo2_resampled = spo2_signal
        
        # Calculate baseline using moving window
        window_size = 120  # 2 minutes
        baseline = pd.Series(spo2_resampled).rolling(
            window=window_size, center=True, min_periods=1
        ).median().values
        
        # Detect desaturations
        desaturations = []
        in_desat = False
        desat_start = 0
        min_duration = 10  # seconds
        
        for i in range(len(spo2_resampled)):
            drop = baseline[i] - spo2_resampled[i]
            
            if drop >= threshold and not in_desat:
                in_desat = True
                desat_start = i
            elif drop < threshold and in_desat:
                duration = i - desat_start
                if duration >= min_duration:
                    desaturations.append((desat_start, i))
                in_desat = False
        
        return desaturations
    
    def detect_arousals(self, eeg_signal: np.ndarray) -> List:
        """
        Detect arousals from EEG signal.
        Simplified detection based on frequency shift.
        """
        # Calculate spectral power in different bands
        nperseg = int(self.sfreq * 4)  # 4-second windows
        noverlap = int(nperseg * 0.5)
        
        freqs, times, Sxx = signal.spectrogram(
            eeg_signal, self.sfreq, nperseg=nperseg, noverlap=noverlap
        )
        
        # Alpha band (8-12 Hz) and Beta band (12-30 Hz)
        alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
        beta_idx = np.where((freqs >= 12) & (freqs <= 30))[0]
        
        alpha_power = np.mean(Sxx[alpha_idx, :], axis=0)
        beta_power = np.mean(Sxx[beta_idx, :], axis=0)
        
        # Detect sudden increases in alpha/beta power
        alpha_threshold = np.percentile(alpha_power, 75)
        beta_threshold = np.percentile(beta_power, 75)
        
        arousals = []
        min_duration = 3  # seconds
        min_gap = 10  # seconds between arousals
        
        arousal_times = times[
            (alpha_power > alpha_threshold) | (beta_power > beta_threshold)
        ]
        
        if len(arousal_times) > 0:
            current_start = arousal_times[0]
            for i in range(1, len(arousal_times)):
                if arousal_times[i] - arousal_times[i-1] > min_gap:
                    if arousal_times[i-1] - current_start >= min_duration:
                        arousals.append((current_start, arousal_times[i-1]))
                    current_start = arousal_times[i]
        
        return arousals
    
    def print_available_channels(self):
        """Print all available channels in the EDF file."""
        print("\n" + "="*60)
        print("AVAILABLE CHANNELS IN EDF FILE:")
        print("="*60)
        for i, ch in enumerate(self.raw.ch_names, 1):
            print(f"{i:3d}. {ch}")
        print("="*60 + "\n")
    
    def calculate_sleep_metrics(self, verbose: bool = True) -> Dict:
        """Calculate all sleep metrics."""
        metrics = {}
        
        if verbose:
            self.print_available_channels()
        
        # Sleep stage metrics
        sleep_stages = self.hypnogram
        total_epochs = len(sleep_stages)
        
        # Total sleep time (all sleep stages)
        tst_epochs = np.sum((sleep_stages >= 1) & (sleep_stages <= 4))
        tst_minutes = (tst_epochs * self.epoch_length) / 60
        tst_hours = tst_minutes / 60
        
        # Time in bed
        tib_minutes = (total_epochs * self.epoch_length) / 60
        
        # Sleep latency (time to first sleep epoch)
        sleep_onset_idx = np.where((sleep_stages >= 1) & (sleep_stages <= 4))[0]
        sleep_latency = 0
        if len(sleep_onset_idx) > 0:
            sleep_latency = (sleep_onset_idx[0] * self.epoch_length) / 60
        
        # Wake after sleep onset (WASO)
        sleep_indices = np.where((sleep_stages >= 1) & (sleep_stages <= 4))[0]
        if len(sleep_indices) > 0:
            first_sleep = sleep_indices[0]
            last_sleep = sleep_indices[-1]
            # Count only wake epochs between sleep onset and final awakening
            waso_epochs = np.sum(sleep_stages[first_sleep:last_sleep+1] == 0)
            waso_minutes = (waso_epochs * self.epoch_length) / 60
        else:
            waso_minutes = 0

        
        # Sleep efficiency
        sleep_efficiency = (tst_minutes / tib_minutes * 100) if tib_minutes > 0 else 0
        
        metrics['sleep_latency_min'] = sleep_latency
        metrics['waso_min'] = waso_minutes
        metrics['total_sleep_time_min'] = tst_minutes
        metrics['sleep_efficiency_pct'] = sleep_efficiency
        
        # Respiratory metrics (AHI, RDI)
        resp_patterns = ['resp', 'airflow', 'nasal', 'thorax', 'abdomen', 'flow', 
                        'pneumo', 'effort', 'chest', 'abdo', 'cannula', 'pressure']
        resp_signal, resp_ch = self.get_channel_data(resp_patterns)
        
        if resp_signal is not None:
            print(f"✓ Found respiratory channel: {resp_ch}")
            
            # SpO2 metrics
            spo2_patterns = ['spo2', 'sao2', 'oxygen', 'o2', 'sat', 'pulse']
            spo2_signal, spo2_ch = self.get_channel_data(spo2_patterns)
            
            if spo2_signal is not None:
                print(f"✓ Found SpO2 channel: {spo2_ch}")
            else:
                print("✗ No SpO2 channel found")
            
            print("Detecting apneas and hypopneas...")
            events = self.detect_apneas_hypopneas(resp_signal, spo2_signal)
            
            # AHI (Apnea-Hypopnea Index)
            ahi = (events['total_events'] / tst_hours) if tst_hours > 0 else 0
            metrics['ahi'] = ahi
            metrics['total_apneas'] = len(events['apneas'])
            metrics['total_hypopneas'] = len(events['hypopneas'])
            
            if spo2_signal is not None:
                # Clean SpO2 signal (remove invalid values)
                valid_spo2 = spo2_signal[(spo2_signal >= 70) & (spo2_signal <= 100)]
                
                if len(valid_spo2) > 0:
                    metrics['min_spo2'] = np.min(valid_spo2)
                    metrics['mean_spo2'] = np.mean(valid_spo2)
                    
                    print("Detecting oxygen desaturations...")
                    # ODI (Oxygen Desaturation Index)
                    desaturations = self.detect_desaturations(spo2_signal)
                    odi = (len(desaturations) / tst_hours) if tst_hours > 0 else 0
                    metrics['odi'] = odi
                    metrics['total_desaturations'] = len(desaturations)
                else:
                    metrics['min_spo2'] = None
                    metrics['mean_spo2'] = None
                    metrics['odi'] = None
                    print("✗ No valid SpO2 data (values outside 70-100% range)")
            
            # RDI (Respiratory Disturbance Index) - includes RERAs
            # For simplification, RDI ≈ AHI + arousal-related events
            metrics['rdi'] = ahi  # Simplified (would need RERA detection)
        else:
            print("✗ No respiratory channel found")
            print("   Searched for patterns:", resp_patterns)
        
        # Arousal Index
        eeg_patterns = ['eeg', 'c3', 'c4', 'f3', 'f4', 'c3-m2', 'c4-m1', 'fpz', 'cz', 'oz']
        eeg_signal, eeg_ch = self.get_channel_data(eeg_patterns)
        
        if eeg_signal is not None:
            print(f"✓ Found EEG channel: {eeg_ch}")
            print("Detecting arousals...")
            arousals = self.detect_arousals(eeg_signal)
            arousal_index = (len(arousals) / tst_hours) if tst_hours > 0 else 0
            metrics['arousal_index'] = arousal_index
            metrics['total_arousals'] = len(arousals)
        else:
            print("✗ No EEG channel found")
        
        # Stage percentages
        if tst_epochs > 0:
            metrics['n1_pct'] = np.sum(sleep_stages == 1) / tst_epochs * 100
            metrics['n2_pct'] = np.sum(sleep_stages == 2) / tst_epochs * 100
            metrics['n3_pct'] = np.sum(sleep_stages == 3) / tst_epochs * 100
            metrics['rem_pct'] = np.sum(sleep_stages == 4) / tst_epochs * 100
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """Print a formatted sleep metrics report."""
        print("\n" + "=" * 60)
        print("SLEEP METRICS REPORT")
        print("=" * 60)
        
        print("\n--- SLEEP ARCHITECTURE ---")
        print(f"Sleep Latency: {metrics.get('sleep_latency_min', 'N/A'):.1f} minutes")
        print(f"Total Sleep Time: {metrics.get('total_sleep_time_min', 'N/A'):.1f} minutes")
        print(f"Wake After Sleep Onset (WASO): {metrics.get('waso_min', 'N/A'):.1f} minutes")
        print(f"Sleep Efficiency: {metrics.get('sleep_efficiency_pct', 'N/A'):.1f}%")
        
        if 'n1_pct' in metrics:
            print(f"\nSleep Stage Distribution:")
            print(f"  N1: {metrics.get('n1_pct', 0):.1f}%")
            print(f"  N2: {metrics.get('n2_pct', 0):.1f}%")
            print(f"  N3: {metrics.get('n3_pct', 0):.1f}%")
            print(f"  REM: {metrics.get('rem_pct', 0):.1f}%")
        
        print("\n--- RESPIRATORY METRICS ---")
        if 'ahi' in metrics:
            print(f"AHI (Apnea-Hypopnea Index): {metrics['ahi']:.1f} events/hour")
            print(f"  Total Apneas: {metrics.get('total_apneas', 'N/A')}")
            print(f"  Total Hypopneas: {metrics.get('total_hypopneas', 'N/A')}")
            print(f"RDI (Respiratory Disturbance Index): {metrics.get('rdi', 'N/A'):.1f} events/hour")
            
            # Severity classification
            ahi_val = metrics['ahi']
            if ahi_val < 5:
                severity = "Normal"
            elif ahi_val < 15:
                severity = "Mild OSA"
            elif ahi_val < 30:
                severity = "Moderate OSA"
            else:
                severity = "Severe OSA"
            print(f"  Severity: {severity}")
        else:
            print("No respiratory data available - missing respiratory channel")
        
        print("\n--- OXYGEN SATURATION ---")
        if metrics.get('mean_spo2') is not None:
            print(f"Mean SpO2: {metrics['mean_spo2']:.1f}%")
            print(f"Minimum SpO2: {metrics['min_spo2']:.1f}%")
        else:
            print("No SpO2 data available - missing SpO2 channel")
            
        if 'odi' in metrics and metrics['odi'] is not None:
            print(f"ODI (Oxygen Desaturation Index): {metrics['odi']:.1f} events/hour")
            print(f"  Total Desaturations (≥3%): {metrics.get('total_desaturations', 'N/A')}")
        
        print("\n--- AROUSAL METRICS ---")
        if 'arousal_index' in metrics:
            print(f"Arousal Index: {metrics['arousal_index']:.1f} arousals/hour")
            print(f"  Total Arousals: {metrics.get('total_arousals', 'N/A')}")
        else:
            print("No arousal data available - missing EEG channel")
        
        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    # Example 1: Load from XML file
    edf_file = r'C:\Users\haris\Downloads\shhs1-200001.edf'
    xml_file = r'C:\Users\haris\Downloads\shhs1-200001-nsrr.xml'  # XML annotation file
    
    try:
        # Option 1: Using XML file for sleep stages
        calculator = SleepMetricsCalculator(
            edf_path=edf_file,
            xml_path=xml_file,
            epoch_length=30
        )
        
        print(f"Loaded {len(calculator.hypnogram)} epochs from XML file")
        print(f"Sleep stages distribution:")
        unique, counts = np.unique(calculator.hypnogram, return_counts=True)
        stage_names = {-1: 'Unknown', 0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
        for stage, count in zip(unique, counts):
            print(f"  {stage_names.get(stage, 'Unknown')}: {count} epochs")
        print()
        
        metrics = calculator.calculate_sleep_metrics()
        calculator.print_report(metrics)
        
        # Save metrics to file
        import json
        with open('sleep_metrics.json', 'w') as f:
            # Convert numpy types to native Python types
            metrics_serializable = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items() if v is not None
            }
            json.dump(metrics_serializable, f, indent=2)
        print("\nMetrics saved to sleep_metrics.json")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("\n=== EXAMPLE XML FORMATS ===\n")
        
        print("Example 1 - NSRR/PSG Format:")
        print("""<?xml version="1.0"?>
<PSGAnnotation>
  <ScoredEvents>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>Wake|0</EventConcept>
      <Start>0</Start>
      <Duration>30</Duration>
    </ScoredEvent>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>NREM1|1</EventConcept>
      <Start>30</Start>
      <Duration>30</Duration>
    </ScoredEvent>
    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>NREM2|2</EventConcept>
      <Start>60</Start>
      <Duration>30</Duration>
    </ScoredEvent>
  </ScoredEvents>
</PSGAnnotation>""")
        
        print("\n\nExample 2 - Simple Format:")
        print("""<?xml version="1.0"?>
<SleepStudy>
  <ScoredEvent>
    <Stage>Wake</Stage>
    <Start>0</Start>
    <Duration>30</Duration>
  </ScoredEvent>
  <ScoredEvent>
    <Stage>N1</Stage>
    <Start>30</Start>
    <Duration>30</Duration>
  </ScoredEvent>
  <ScoredEvent>
    <Stage>N2</Stage>
    <Start>60</Start>
    <Duration>30</Duration>
  </ScoredEvent>
  <ScoredEvent>
    <Stage>REM</Stage>
    <Start>90</Start>
    <Duration>30</Duration>
  </ScoredEvent>
</SleepStudy>""")
        
        print("\n\nTo use this script:")
        print("1. Replace 'sleep_study.edf' with your EDF file path")
        print("2. Replace 'sleep_stages.xml' with your XML annotation file")
        print("3. Ensure MNE-Python is installed: pip install mne")
        print("\nSupported XML formats:")
        print("- NSRR (National Sleep Research Resource)")
        print("- Compumedics Profusion")
        print("- Generic scored events format")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure your XML file contains sleep stage annotations.")
    
    # Example 2: Using numpy array directly (original method)
    print("\n" + "="*60)
    print("You can also provide sleep stages directly as a numpy array:")
    print("="*60)
    print("""
hypnogram = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 2, 2, 4, 4, 2, 0, 0])

calculator = SleepMetricsCalculator(
    edf_path="sleep_study.edf",
    hypnogram=hypnogram,
    epoch_length=30
)
""")