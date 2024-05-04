import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pywt

def add_noise_to_stock_data(file_path):
    # Load the closing 1hr stock prices from the Excel file
    data = pd.read_excel(file_path, usecols=['time', 'TSLA close', 'WMT close', 'JNJ close', 'INTC close', 'AAPL close'])
    
    # Create a data frame to store noisy data
    noisy_data = data.copy()
    
    # Columns containing the stock close prices
    stock_columns = ['TSLA close', 'WMT close', 'JNJ close', 'INTC close', 'AAPL close']
    
    # Apply multiple of Gaussian noise to each stock column and plot
    for column in stock_columns:
        # Creating Gaussian noise
        noise = np.random.normal(0, 1, len(data))
        
        # Applying noise to the stock prices
        noisy_data[column] = data[column] * (1 + data[column].shift(1).fillna(data[column].iloc[0]) * (1/10000) * noise)
        
        # Create figure with two subplots, one above the other
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot original data on first subplot
        axs[0].plot(data[column], label='Original')
        axs[0].set_title(f"{column} - Original")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot noisy data on second subplot
        axs[1].plot(noisy_data[column], label='With Noise', color='red')
        axs[1].set_title(f"{column} - Noisy")
        axs[1].set_xlabel('Number of Data Points (Hours)')
        axs[1].set_ylabel('Price')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return noisy_data, data

def haar_denoising(data, threshold, method='hard'):
    # Ensure no NaN values are present
    if data.isna().any():
        data = data.fillna(method='ffill')

    # Haar wavelet decomposition
    coeffs = pywt.wavedec(data, 'haar', mode='per')
    
    # Select thresholding function
    threshold_mode = 'hard' if method == 'hard' else 'soft'
    
    # Apply thresholding to coefficients
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode=threshold_mode) for c in coeffs[1:]]
    
    # Reconstruct signal from thresholded coefficients
    denoised_data = pywt.waverec(coeffs, 'haar', mode='per')
    
    return denoised_data

def daubechies_denoising(data, threshold, order, method='soft'):
    # Ensure no NaN values are present
    if data.isna().any():
        data = data.fillna(method='ffill')

    # Specify wavelet order
    wavelet = f'db{order}'

    # Daubechies wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric')

    # Select thresholding function
    threshold_mode = 'hard' if method == 'hard' else 'soft'
    
    # Apply thresholding to coefficients
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode=threshold_mode) for c in coeffs[1:]]

    # Reconstruct signal from thresholded coefficients
    denoised_data = pywt.waverec(coeffs, wavelet, mode='symmetric')

    return denoised_data

def symlet_denoising(data, threshold, order, method='soft'):
    # Ensure no NaN values are present
    if data.isna().any():
        data = data.fillna(method='ffill')

    # Specify the wavelet order
    wavelet = f'sym{order}'

    # Symlet wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric')

    # Select thresholding function
    threshold_mode = 'hard' if method == 'hard' else 'soft'
    
    # Apply thresholding to coefficients
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode=threshold_mode) for c in coeffs[1:]]

    # Reconstruct signal from thresholded coefficients
    denoised_data = pywt.waverec(coeffs, wavelet, mode='symmetric')

    return denoised_data

def calculate_mse(original, denoised):
    # Calculate MSE between original and denoised data
    return np.mean((denoised - original) ** 2)

def calculate_snr(original, denoised):
    # Calculate signal-to-noise ratio (dB) between original and denoised data
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((denoised  - original) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def optimize_denoising(data, original, wavelet='haar', max_order=5, threshold_range=(1, 10), method='soft'):
    # After inputting noisy data, original data, wavelet type, and wavelet type, this function loops through 
    # different thresholds and order values to determine which parameters 

    # Initialize variables
    # Negative infinity is used for initializing SNR to ensure that any real SNR calculated is larger
    # Similarly, positive infinity is used for initializing MSE to ensure that any real MSE calculated is smaller
    best_snr = -np.inf
    best_mse = np.inf
    best_params = {}
    
    # Nested for loops to go through all order and threshold combinations
    for order in range(1, max_order):
        for threshold in np.linspace(*threshold_range, num=10):

            # Construct wavelet using different pre-defined functions based on type and order input into optimize_denoising
            wavelet_name = f'{wavelet}{order}' if wavelet in ['db', 'sym'] else wavelet
            
            # Denoise data based on which type of wavelet is input
            if wavelet == 'haar':
                denoised_data = haar_denoising(data, threshold, method)
            elif wavelet in ['db', 'sym']:
                denoised_data = daubechies_denoising(data, threshold, order + 1, method) if wavelet == 'db' else symlet_denoising(data, threshold, order + 1, method)
            
            # Calculate MSE and SNR
            mse = calculate_mse(original, denoised_data)
            snr = calculate_snr(original, denoised_data)
            
            # Check based on MSE and SNR combined if selected parameters are best so far
            if snr > best_snr:
                best_snr = snr
                best_mse = mse
                best_params = {'wavelet': wavelet_name, 'threshold': threshold, 'method': method, 'order': order}
    
    return best_params, best_snr, best_mse

file_path = r'C:\Users\willk\OneDrive\Desktop\AMS522 HW7\AMS_522_Project_1hr_Stock_Data.xlsx'
noisy_data, data = add_noise_to_stock_data(file_path)

# Example denoising for illustrative purposes
denoised_WMT = symlet_denoising(noisy_data['WMT close'], 10, 5, method='soft')
plt.plot(denoised_WMT, label='Denoised WMT', color='red')
plt.title(f"WMT - Denoised")
plt.xlabel('Number of Data Points (Hours)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# optimize_denoising function
for i in ['TSLA','WMT','JNJ','INTC','AAPL']:
    best_params, best_snr, best_mse = optimize_denoising(noisy_data[f'{i} close'], data[f'{i} close'], wavelet='db', max_order=19, threshold_range=(2, 20), method='hard')
    print(f"Best Parameters - {i}: {best_params}")
    print(f"Best SNR - {i}: {best_snr} dB")
    print(f"Best MSE - {i}: {best_mse}")

# Plot of TSLA - Denoised and Original
original_TSLA = data['TSLA close']
denoised_TSLA_np = symlet_denoising(noisy_data['TSLA close'], 8.0, 10, method='soft')
denoised_TSLA = pd.Series(denoised_TSLA_np, index=original_TSLA.index)
log_returns_original_TSLA = np.log(original_TSLA / original_TSLA.shift(1))
log_returns_denoised_TSLA = np.log(denoised_TSLA / denoised_TSLA.shift(1))
original_volatility_TSLA = log_returns_original_TSLA.std()
denoised_volatility_TSLA = log_returns_denoised_TSLA.std()
annualized_volatility_original_TSLA = original_volatility_TSLA * np.sqrt(252*6.5)
annualized_volatility_denoised_TSLA = denoised_volatility_TSLA * np.sqrt(252*6.5)
print("Original TSLA Volatility:", annualized_volatility_original_TSLA)
print("Denoised TSLA Volatility:", annualized_volatility_denoised_TSLA)
plt.plot(original_TSLA.index, original_TSLA, label='Original TSLA', color='blue')
plt.plot(original_TSLA.index, denoised_TSLA, label='Denoised TSLA', color='red')
plt.title(f"TSLA - Original vs Denoised")
plt.xlabel('Number of Data Points (Hours)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot of WMT - Denoised and Original
original_WMT = data['WMT close']
denoised_WMT_np = symlet_denoising(noisy_data['WMT close'], 2.0, 11, method='hard')
denoised_WMT = pd.Series(denoised_WMT_np, index=original_WMT.index)
log_returns_original_WMT = np.log(original_WMT / original_WMT.shift(1))
log_returns_denoised_WMT = np.log(denoised_WMT / denoised_WMT.shift(1))
original_volatility_WMT = log_returns_original_WMT.std()
denoised_volatility_WMT = log_returns_denoised_WMT.std()
annualized_volatility_original_WMT = original_volatility_WMT * np.sqrt(252*6.5)
annualized_volatility_denoised_WMT = denoised_volatility_WMT * np.sqrt(252*6.5)
print("Original WMT Volatility:", annualized_volatility_original_WMT)
print("Denoised WMT Volatility:", annualized_volatility_denoised_WMT)
plt.plot(original_WMT.index, original_WMT, label='Original WMT', color='blue')
plt.plot(original_WMT.index, denoised_WMT, label='Denoised WMT', color='red')
plt.title(f"WMT - Original vs Denoised")
plt.xlabel('Number of Data Points (Hours)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot of JNJ - Denoised and Original
original_JNJ = data['JNJ close']
denoised_JNJ_np = symlet_denoising(noisy_data['JNJ close'], 8.0, 16, method='hard')
denoised_JNJ = pd.Series(denoised_JNJ_np, index=original_JNJ.index)
log_returns_original_JNJ = np.log(original_JNJ / original_JNJ.shift(1))
log_returns_denoised_JNJ = np.log(denoised_JNJ / denoised_JNJ.shift(1))
original_volatility_JNJ = log_returns_original_JNJ.std()
denoised_volatility_JNJ = log_returns_denoised_JNJ.std()
annualized_volatility_original_JNJ = original_volatility_JNJ * np.sqrt(252*6.5)
annualized_volatility_denoised_JNJ = denoised_volatility_JNJ * np.sqrt(252*6.5)
print("Original JNJ Volatility:", annualized_volatility_original_JNJ)
print("Denoised JNJ Volatility:", annualized_volatility_denoised_JNJ)
plt.plot(original_JNJ.index, original_JNJ, label='Original JNJ', color='blue')
plt.plot(original_JNJ.index, denoised_JNJ, label='Denoised JNJ', color='red')
plt.title(f"JNJ - Original vs Denoised")
plt.xlabel('Number of Data Points (Hours)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot of INTC - Denoised and Original
original_INTC = data['INTC close']
denoised_INTC_np = symlet_denoising(noisy_data['INTC close'], 2.0, 17, method='hard')
denoised_INTC = pd.Series(denoised_INTC_np, index=original_INTC.index)
log_returns_original_INTC = np.log(original_INTC / original_INTC.shift(1))
log_returns_denoised_INTC = np.log(denoised_INTC / denoised_INTC.shift(1))
original_volatility_INTC = log_returns_original_INTC.std()
denoised_volatility_INTC = log_returns_denoised_INTC.std()
annualized_volatility_original_INTC = original_volatility_INTC * np.sqrt(252*6.5)
annualized_volatility_denoised_INTC = denoised_volatility_INTC * np.sqrt(252*6.5)
print("Original INTC Volatility:", annualized_volatility_original_INTC)
print("Denoised INTC Volatility:", annualized_volatility_denoised_INTC)
plt.plot(original_INTC.index, original_INTC, label='Original INTC', color='blue')
plt.plot(original_INTC.index, denoised_INTC, label='Denoised INTC', color='red')
plt.title(f"INTC - Original vs Denoised")
plt.xlabel('Number of Data Points (Hours)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot of AAPL - Denoised and Original
original_AAPL = data['AAPL close']
denoised_AAPL_np = daubechies_denoising(noisy_data['AAPL close'], 8.0, 4, method='hard')
denoised_AAPL = pd.Series(denoised_AAPL_np, index=original_AAPL.index)
log_returns_original_AAPL = np.log(original_AAPL / original_AAPL.shift(1))
log_returns_denoised_AAPL = np.log(denoised_AAPL / denoised_AAPL.shift(1))
original_volatility_AAPL = log_returns_original_AAPL.std()
denoised_volatility_AAPL = log_returns_denoised_AAPL.std()
annualized_volatility_original_AAPL = original_volatility_AAPL * np.sqrt(252*6.5)
annualized_volatility_denoised_AAPL = denoised_volatility_AAPL * np.sqrt(252*6.5)
print("Original AAPL Volatility:", annualized_volatility_original_AAPL)
print("Denoised AAPL Volatility:", annualized_volatility_denoised_AAPL)
plt.plot(original_AAPL.index, original_AAPL, label='Original AAPL', color='blue')
plt.plot(original_AAPL.index, denoised_AAPL, label='Denoised AAPL', color='red')
plt.title(f"AAPL - Original vs Denoised")
plt.xlabel('Number of Data Points (Hours)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()