from .biometric_features_extractor import BiometricFeaturesExtractor
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk

class Biometricfeat(BiometricFeaturesExtractor):
    def __init__(self):
        self._csvpath = "datasets/biometrics.csv"
        
    def read_file(self, path):
        with open(path, 'rb') as file:
            #Carica l'oggetto dal file pickle
            data = pickle.load(file, encoding='latin1')
        return data
    
    @staticmethod
    def fenestration(signal, campionamento):

        if signal.shape[1] == 1:
            return signal.reshape(-1,campionamento)
        else: 
            return signal.reshape(-1, campionamento, signal.shape[1])
        
    def extract_features(self, signal, sampling_rate):
        pass
    
    # calcola il primo e l'ultimo indice corrispondente ad un valore emozionale costante
    # return - np.array con (index_start, index_end, valore)
    @staticmethod
    def get_emotions_index(labels):
        indexes = []
        for i, row in enumerate(labels):
            unique_value = np.unique(row)
            if len(unique_value) == 1:
                indexes.append([i,unique_value[0]])
        
        if indexes != 0:
            emotion_index = []
            start = indexes[0][0]
            last = indexes[0][0]
            value = indexes[0][1]
            for i, part in enumerate(indexes[1:]):
                if part[1] != value or i == len(indexes[1:])-1 :
                    emotion_index.append([start, last+1, value])
                    start = part[0]
                    value = part[1]
                
                last = part[0]
            if emotion_index != 0:
                for i, row in enumerate(emotion_index):
                    if row[2] in [5,6,7]:
                        emotion_index.pop(i)
                return np.array(emotion_index)
        
        return np.empty([1,1])
    
    @staticmethod
    def create_data_frame(features, tag, emotions):
        df = pd.DataFrame()
        for signal_type in features:
            for feature_name, value in signal_type:
                df[feature_name] = [value]
        df['Emotion_Code'] = tag
        df['Emotion'] = emotions[tag]
        return df
    
    # estrattore di features BVP(Blood Volume Pulse) utilizzando la libreria neuralkit2 a supporto:
    # dominio t: HR, HRV: RMSSD, SDNN, pNN50
    # dominio f: LF, HF, LF/HF ratio
    # parametri non lineari: SD1, SD2, Poincaré Ratio (SD1/SD2 Entropy Measures (es. Sample Entropy, Approximate Entropy), Fractal Dimension, Detrended Fluctuation Analysis (DFA)       
    @staticmethod
    def extract_bvp_features(signal, sampling_rate):
        
        bvp_signals, info = nk.ppg_process(signal.flatten(), sampling_rate=sampling_rate)
        # segnale pulito
        bvp_clean_signal = bvp_signals.get('PPG_Clean')
        bvp_clean_signal_mean = np.nanmean(bvp_clean_signal.values)
        bvp_clean_signal_std = np.nanstd(bvp_clean_signal.values)
        # ampiezza dei picchi (volume di sangue pulsato nei vasi)
        #peaks = bvp_signals.get('PPG_Peaks')
        peaks = nk.ppg_findpeaks(bvp_clean_signal, sampling_rate)['PPG_Peaks']
        peak_amplitude = bvp_clean_signal.loc[peaks].values
        peak_amplitude_mean = np.nanmean(peak_amplitude)
        peak_amplitude_std = np.nanstd(peak_amplitude)
        # distnza temporale tra due picchi successivi nel segnale bvp
        ibi = np.diff(peaks) / sampling_rate  # IBI in secondi
        ibi_mean = np.nanmean(ibi)
        ibi_std = np.nanstd(ibi)
        # freq. cardiaca
        hr = bvp_signals.get('PPG_Rate')
        hr_mean = np.nanmean(hr.values)
        hr_std = np.nanstd(hr.values)
        # parametri variabilità freq. cardiaca (tempo)
        hrv_time = nk.hrv_time(peaks, sampling_rate=64)
        sdnn = hrv_time['HRV_SDNN']
        rmssd = hrv_time['HRV_RMSSD']
        pnn50 = hrv_time['HRV_pNN50']
        # parametri variabilità freq. cardiaca (frequenza)
        hrv_frequency = nk.hrv_frequency(peaks, sampling_rate=64)
        lf = hrv_frequency['HRV_LF']
        hf = hrv_frequency['HRV_HF']
        lfhf = hrv_frequency['HRV_LFHF']
        # parametri non lineari
        hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=64)
        sd1 = hrv_nonlinear['HRV_SD1']
        sd2 = hrv_nonlinear['HRV_SD2']
        sd1sd2 = hrv_nonlinear['HRV_SD1SD2']
        sampEn = hrv_nonlinear['HRV_SampEn']
        apEn = hrv_nonlinear['HRV_ApEn']
        dfa = hrv_nonlinear['HRV_DFA_alpha1']
        hfd = hrv_nonlinear['HRV_HFD']
        
        features = [
            ['BVP_Clean_Signal_Mean', bvp_clean_signal_mean],
            ['BVP_Clean_Signal_Std', bvp_clean_signal_std],  
            ['BVP_Peak_Amplitude_Mean', peak_amplitude_mean],
            ['BVP_Peak_Amplitude_Std', peak_amplitude_std],
            ['BVP_IBI_Mean', ibi_mean],
            ['BVP_IBI_Std', ibi_std],
            ['BVP_HR_Mean', hr_mean],
            ['BVP_HR_Std', hr_std],
            ['BVP_SDNN', sdnn.values[0]],
            ['BVP_RMSSD', rmssd.values[0]],
            ['BVP_PNN50', pnn50.values[0]],
            ['BVP_LF', lf.values[0]],
            ['BVP_HF', hf.values[0]],
            ['BVP_LFHF', lfhf.values[0]],
            ['BVP_SD1', sd1.values[0]],
            ['BVP_SD2', sd2.values[0]],
            ['BVP_SD1SD2', sd1sd2.values[0]],
            ['BVP_SampEn', sampEn.values[0]],
            ['BVP_ApEn', apEn.values[0]],
            ['BVP_DFA', dfa.values[0]],
            ['BVP_HFD', hfd.values[0]]
        ]
        return features
    
    # parametri EDA da considerare:
    # SCL (Skin Conductance Level), SCR (Skin Conductance Response), Ampiezza delle SCR, Frequenza delle SCR, Tempo di picco della SCR
    @staticmethod
    def extract_eda_features(signal, sampling_rate): 
          
        eda_signals, info = nk.eda_process(signal.flatten(), sampling_rate=sampling_rate)
        eda_clean_signal = eda_signals.get('EDA_Clean')
        eda_clean_signal_mean = np.nanmean(eda_clean_signal.values)
        eda_clean_signal_std = np.nanstd(eda_clean_signal.values)
        scl = eda_signals.get('EDA_Tonic')
        scl_mean = np.nanmean(scl.values)
        scl_std = np.nanstd(scl.values)
        scr = eda_signals.get('EDA_Phasic')
        scr_mean = np.nanmean(scr.values)
        scr_std = np.nanstd(scr.values)
        peaks = nk.eda_findpeaks(scr.values, sampling_rate)
        scr_peaks_count = len(peaks['SCR_Peaks'])
        scr_peaks_amplitude = peaks['SCR_Height']
        scr_peaks_amplitude_mean = np.nanmean(scr_peaks_amplitude)
        scr_peaks_amplitude_max = np.max(scr_peaks_amplitude)
        scr_peaks_amplitude_std = np.nanstd(scr_peaks_amplitude)
        peaks_interval = np.diff(peaks['SCR_Peaks'])/sampling_rate
        features = [
            ['EDA_Clean_Signal_Mean', eda_clean_signal_mean],
            ['EDA_Clean_Signal_Std', eda_clean_signal_std],  
            ['EDA_SCL_Mean', scl_mean],
            ['EDA_SCL_Std', scl_std],
            ['EDA_SCR_Mean', scr_mean],
            ['EDA_SCR_Std', scr_std],
            ['EDA_SCR_Peaks_Count', scr_peaks_count],
            ['EDA_SCR_Peaks_Amplitude_Mean', scr_peaks_amplitude_mean],
            ['EDA_SCR_Peaks_Amplitude_Max', scr_peaks_amplitude_max],
            ['EDA_SCR_Peaks_Amplitude_Std', scr_peaks_amplitude_std],
            ['EDA_Peaks_Interval_Mean', np.nanmean(peaks_interval) if len(peaks_interval) > 0 else np.nan]
        ]
        return features
    
    # parametri TEMP da considerare:
    # Temperatura Corporea, Variazione della Temperatura, Velocità di Cambiamento,Media Mobile, Frequenza di Misurazione, Picchi e Valli
    @staticmethod
    def extract_temp_features(signal, sampling_rate):
        
        temp_clean = nk.signal_smooth(signal.flatten(), size=15)
        # VALORI STATISTICI: media, mediana, varianza, deviazione standard (su finestre temporali)
        mean = np.nanmean(temp_clean)
        median = np.nanmedian(temp_clean)
        variance = np.nanvar(temp_clean)
        dev_std = np.nanstd(temp_clean)
        min = np.min(temp_clean)
        max = np.max(temp_clean)
        range_temp = max-min
        # PARAMETRI DI VARIAZIONE RAPIDA: Delta tra punti consecutivi, picchi, numero di picchi...
        delta = np.diff(temp_clean)
        # Peaks
        positive_peaks_info = nk.signal_findpeaks(temp_clean, relative_height_min=max*0.1)
        negative_peaks_info = nk.signal_findpeaks(-temp_clean, relative_height_min=max*0.1)
        # Picchi positivi
        positive_peaks = positive_peaks_info['Peaks']
        positive_peaks_count = len(positive_peaks)
        positive_peaks_amplitude = positive_peaks_info['Height']
        positive_mean_peaks_amplitude = np.nanmean(positive_peaks_amplitude) if len(positive_peaks_amplitude) > 0 else np.nan
        # Picchi negativi
        negative_peaks = negative_peaks_info['Peaks']
        negative_peaks_count = len(negative_peaks)
        negative_peaks_amplitude = -negative_peaks_info['Height']
        negative_mean_peaks_amplitude = np.nanmean(negative_peaks_amplitude) if len(negative_peaks_amplitude) > 0 else np.nan
        # Derivate
        temp_first_derivative = np.gradient(temp_clean)
        temp_second_derivative = np.gradient(temp_first_derivative)
        #trend lineare
        time = np.arange(len(temp_clean))
        slope, intercept = np.polyfit(time, temp_clean, 1)
        linear_trend = slope * time + intercept
        features = [
            ['TEMP_Mean', mean],
            ['TEMP_Median', median],
            ['TEMP_Variance', variance ],
            ['TEMP_Std', dev_std ],
            ['TEMP_Min', min],
            ['TEMP_Max', max],
            ['TEMP_MaxMinRange', range_temp],
            ['TEMP_Delta_Mean', np.nanmean(delta)],
            ['TEMP_Positive_Peaks_Count', positive_peaks_count],
            ['TEMP_Positive_Peaks_Amplitude_Mean', positive_mean_peaks_amplitude],
            ['TEMP_Negative_Peaks_Count', negative_peaks_count],
            ['TEMP_Negative_Peaks_Amplitude_Mean', negative_mean_peaks_amplitude],
            ['TEMP_First_Derivative_Mean', np.nanmean(temp_first_derivative)],
            ['TEMP_First_Derivative_Std', np.nanstd(temp_first_derivative)],
            ['TEMP_Second_Derivative_Mean', np.nanmean(temp_second_derivative)],
            ['TEMP_Second_Derivative_Std', np.nanstd(temp_second_derivative)],
            ['TEMP_Slope', slope],
            ['TEMP_Intercept', intercept]      
        ]
        return features

    # parametri ACC da considerare:
    # Dominio t: Valori medi e deviazione standard, Energia media, Peak-to-peak, Cross-correlation, Quantità di movimento (magnitude vector)
    # Dominio f: Power Spectral Density (PSD), Peak Frequency, Bande di frequenza
    # Feature derivate: Varianza della magnitudo del vettore (X, Y, Z combinati), Root Mean Square (RMS), Jerk (derivata della magnitudo dell'accelerazione)
    @staticmethod
    def extract_acc_features(signal, sampling_rate):
        acc_x, acc_y, acc_z = zip(*[(line[0], line[1], line[2]) for row in signal for line in row])
        acc_x = list(acc_x)
        acc_y = list(acc_y)
        acc_z = list(acc_z)
        acc_x_clean = nk.signal_filter(acc_x, sampling_rate, lowcut=0.5, highcut=10, method="butter")
        acc_y_clean = nk.signal_filter(acc_y, sampling_rate, lowcut=0.5, highcut=10, method="butter")
        acc_z_clean = nk.signal_filter(acc_z, sampling_rate, lowcut=0.5, highcut=10, method="butter")
        # Dominio t
        # Valori medi e deviazione standard
        acc_x_mean = np.nanmean(acc_x_clean)
        acc_y_mean = np.nanmean(acc_y_clean)
        acc_z_mean = np.nanmean(acc_z_clean)

        acc_x_std = np.nanstd(acc_x_clean)
        acc_y_std = np.nanstd(acc_y_clean)
        acc_z_std = np.nanstd(acc_z_clean)

        # Energia media del segnale (intensità del segnale)
        acc_x_energy = np.nanmean(acc_x_clean**2)
        acc_y_energy = np.nanmean(acc_y_clean**2)
        acc_z_energy = np.nanmean(acc_z_clean**2)

        # Peak-to-peak (massimo - minimo) di ogni asse
        acc_x_ptp = np.max(acc_x_clean) - np.min(acc_x_clean)
        acc_y_ptp = np.max(acc_y_clean) - np.min(acc_y_clean)
        acc_z_ptp = np.max(acc_z_clean) - np.min(acc_z_clean)

        #Cross-correlation tra gli assi X-Y, Y-Z e Z-X
        cross_corr_xy = np.correlate(acc_x_clean, acc_y_clean, mode="full")
        cross_corr_yz = np.correlate(acc_y_clean, acc_z_clean, mode="full")
        cross_corr_xz = np.correlate(acc_x_clean, acc_z_clean, mode="full")

        # Quantità di movimento (magnitude vector)
        acc_x_magnitude = np.sqrt(np.sum(acc_x_clean**2))
        acc_y_magnitude = np.sqrt(np.sum(acc_y_clean**2))
        acc_z_magnitude = np.sqrt(np.sum(acc_z_clean**2))

        # Varianza della magnitudo
        #acc_x_var_magnitude = np.nanvar(acc_x_magnitude)
        #acc_y_var_magnitude = np.nanvar(acc_y_magnitude)
        #acc_z_var_magnitude = np.nanvar(acc_z_magnitude)

        # Dominio f
        # Power Spectral Density (PSD) per ciascun asse
        acc_x_psd = nk.signal_psd(acc_x_clean, 32)
        acc_y_psd = nk.signal_psd(acc_y_clean, 32) 
        acc_z_psd = nk.signal_psd(acc_z_clean, 32) 

        # Peak Frequency (frequenza di picco)
        acc_x_peak_freq = acc_x_psd['Frequency'][np.argmax(acc_x_psd['Power'])]
        acc_y_peak_freq = acc_y_psd['Frequency'][np.argmax(acc_y_psd['Power'])]
        acc_z_peak_freq = acc_z_psd['Frequency'][np.argmax(acc_z_psd['Power'])]
        #print(acc_x_peak_freq, acc_y_peak_freq, acc_z_peak_freq)

        # Root Mean Square (RMS) per ogni asse
        acc_x_rms = np.sqrt(np.nanmean(acc_x_clean**2))
        acc_y_rms = np.sqrt(np.nanmean(acc_y_clean**2))
        acc_z_rms = np.sqrt(np.nanmean(acc_y_clean**2))
        
        features = [
            ['ACC_X_Mean', acc_x_mean],
            ['ACC_Y_Mean', acc_y_mean],
            ['ACC_Z_Mean', acc_z_mean],
            ['ACC_X_Std', acc_x_std],
            ['ACC_Y_Std', acc_y_std],
            ['ACC_Z_Std', acc_z_std],
            ['ACC_X_Energy', acc_x_energy],
            ['ACC_Y_Energy', acc_y_energy],
            ['ACC_Z_Energy', acc_z_energy],
            ['ACC_X_PeaktoPeak', acc_x_ptp],
            ['ACC_Y_PeaktoPeak', acc_y_ptp],
            ['ACC_Z_PeaktoPeak', acc_z_ptp],
            ['ACC_Cross_Correlation_XY', np.nanmean(cross_corr_xy)],
            ['ACC_Cross_Correlation_YZ', np.nanmean(cross_corr_yz)],
            ['ACC_Cross_Correlation_XZ', np.nanmean(cross_corr_xz)],
            ['ACC_X_Magnitude', acc_x_magnitude],
            ['ACC_Y_Magnitude', acc_y_magnitude],
            ['ACC_Z_Magnitude', acc_z_magnitude],
            #['ACC_X_Magnitude_Variance', acc_x_var_magnitude],
            #['ACC_Y_Magnitude_Variance', acc_y_var_magnitude],
            #['ACC_Z_Magnitude_Variance', acc_z_var_magnitude],
            ['ACC_X_PSD_Mean', np.nanmean(acc_x_psd['Power'])],
            ['ACC_Y_PSD_Mean', np.nanmean(acc_y_psd['Power'])],
            ['ACC_Z_PSD_Mean', np.nanmean(acc_z_psd['Power'])],
            ['ACC_X_PSD_Peak_Frequency', acc_x_peak_freq],
            ['ACC_Y_PSD_Peak_Frequency', acc_y_peak_freq],
            ['ACC_Z_PSD_Peak_Frequency', acc_z_peak_freq],
            ['ACC_X_RMS', acc_x_rms],
            ['ACC_Y_RMS', acc_y_rms],
            ['ACC_Z_RMS', acc_z_rms]  
            ]
        
        return features
    
    # funzione che estrae le features biometriche, prendendo finestre temporali dei segnali corrispondenti ad emozioni
    # successivamente mette insieme tutte le features in un dataframe pronto per la stampa su file
    # window[0]: indice inziale, window[1]: indice finale, window[2]: etichetta
    @staticmethod
    def extract_biometric_features(windows: np.ndarray, bvp: np.ndarray, eda: np.ndarray, temp: np.ndarray , acc: np.ndarray)->list:
        windows_features = []
        for window in windows:
            bvp_features = Biometricfeat.extract_bvp_features(bvp[window[0]:window[1]], 64)
            eda_features = Biometricfeat.extract_eda_features(eda[window[0]:window[1]], 4)
            temp_features = Biometricfeat.extract_temp_features(temp[window[0]:window[1]], 4)
            acc_features = Biometricfeat.extract_acc_features(acc[window[0]:window[1]], 32)
            windows_features.append({"BVP_Features":bvp_features, "EDA_Features":eda_features, "TEMP_Features":temp_features, "ACC_Features":acc_features, "Emotion_Code":window[2]})
            
        return windows_features
    
    def run(self, file_paths, emotions):
        for path in file_paths:
            data = self.read_file(path)
            #print(data)
            wrist_signals = data['signal']['wrist']
            # segnali separati -> numpy.ndarray type
            tag_raw = data['label'].reshape(-1,1) #trasformo in vettore colonna
            acc_raw = wrist_signals['ACC']
            bvp_raw = wrist_signals['BVP']
            eda_raw = wrist_signals['EDA']
            temp_raw = wrist_signals['TEMP']
            
            tags = Biometricfeat.fenestration(tag_raw, 700)
            bvp = Biometricfeat.fenestration(bvp_raw, 64)
            eda = Biometricfeat.fenestration(eda_raw, 4)
            temp = Biometricfeat.fenestration(temp_raw, 4)
            acc = Biometricfeat.fenestration(acc_raw, 32)
            
            windows = Biometricfeat.get_emotions_index(tags)
            
            windows_features = Biometricfeat.extract_biometric_features(windows, bvp, eda, temp, acc)
            for window_feature in windows_features:
                frame = Biometricfeat.create_data_frame([window_feature["BVP_Features"], window_feature["EDA_Features"], window_feature["TEMP_Features"], window_feature["ACC_Features"]], window_feature["Emotion_Code"], emotions)
                frame.to_csv(self._csvpath, index=False, mode='a')