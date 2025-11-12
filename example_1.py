from packet import ImprovedWaveletPacketJPointDetector
from main import EnhancedSTAnalyzer
import neurokit2 as nk

# Initialize detector and analyzer
j_detector = ImprovedWaveletPacketJPointDetector(sampling_rate=1000)
analyzer = EnhancedSTAnalyzer(sampling_rate=1000)
analyzer.set_j_detector(j_detector)

# Generate or load an ECG beat (example uses NeuroKit2 synthetic data)
ecq = nk.data("ecg_1000hz")
cleaned = nk.ecg_clean(ecq, sampling_rate=1000)
peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=1000)
rpeaks_idx = np.where(peaks["ECG_R_Peaks"] == 1)[0]
beats = nk.ecg_segment(cleaned, rpeaks_idx, sampling_rate=1000)
beat = list(beats.values())[0]["Signal"].values

# Run analysis
results = analyzer.analyze_st_segment_comprehensive(beat)
analyzer.print_st_analysis_results(results)
analyzer.plot_st_analysis(beat, results)