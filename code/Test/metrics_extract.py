import mne
import yasa

# Load PSG (signal) and hypnogram (staging/annotation)
raw = mne.io.read_raw_edf(r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf", preload=True)
hypno = mne.read_annotations(r"C:\PS\Sleep-Staging\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001EC-Hypnogram.edf")
raw.set_annotations(hypno)
print(raw.annotations.onset[:5])
print(raw.annotations.duration[:5])
print(raw.annotations.description[:5])
# Use YASA for sleep summary (assuming hypno follows YASA conventions)
# sleep_stats = yasa.sleep_statistics(hypno, raw.info['sfreq'])
# print(sleep_stats)
