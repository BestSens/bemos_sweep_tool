from typing import Optional
from bone.bone_connect import bone_connect
from math import sin, cos, pi, ceil, log10
import inquirer
import time
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

from scipy.interpolate import interp1d

DAC_SAMPLE_RATE = 4 * 10 ** 6
ADC_SAMPLE_RATE = 10 * 10 ** 6

vga = 1
level = 127.
f_min = 100_000
f_max = 1_000_000

dv_length = 1024

integral_gates = [
	(None, None)
]

def hann_window(series: list) -> list:
	"""Apply Von Hann Window to the given list

	Parameters
	----------
	series: list
		List of ints or floats to apply the window to

	Returns
	--------
	list of floats
		data with applied Von Hann Window

	Raises
	------
	ValueError
		if length of list is <= 1
	TypeError
		if list elements are not numbers
	"""
	if len(series) <= 1:
		raise ValueError('List has to be longer than 1 element')
	if all(isinstance(value, (int, float)) for value in series):
		return [value * 0.5 * (1 - cos(2 * pi * index / (len(series) - 1))) for index, value in enumerate(series)]
	else:
		raise TypeError("List elements must be numbers")


def generate_sin_pulse(frequency=500000,
					   number_of_cycles=5,
					   sample_rate=4 * 10 ** 6,
					   enable_hann=True,
					   fill_zeros_to=None) -> list:
	"""Generate sin pulse of specified frequency, number of cycles, sample_rate, optional Von Hann Window and
	optional zero padding.

	Parameters
	----------
	frequency: float
		Frequency in Hz of the sin pulse. Default: 500000
	number_of_cycles: int
		Number of periods of the sin pulse. Default: 5
	sample_rate: int
		Sample rate of the returned sin pulse. Default: 4M
	enable_hann: bool
		Apply Von Hann Window Function to sin pulse. Default: True
	fill_zeros_to: None, int
		if given, the pulse is filled with zeros to given number of samples

	Returns
	-------
	list of floats
		List of values of the sin pulse

	Raises
	------
	ValueError
		if length of pulse is bigger than fill_zeros_to
		if frequnxy is <= 0
		if number_of_cylces is <=0
		if sample_rate is <= 0
		if sample_rate/2 is < frequency (Shannon-Nyquist-Theorem)
	"""
	if frequency <= 0:
		raise ValueError('frequency has to be greater than zero')
	if number_of_cycles <= 0:
		raise ValueError('number_of_cycles has to be greater than zero')
	if sample_rate <= 0:
		raise ValueError('sample_rate has to be greater than zero')
	if sample_rate / 2 < frequency:
		raise ValueError('sample rate has to be at least twice as high as the frequency (Shannon-Nyquist-Theorem)')

	pulse = [sin(frequency * 2 * pi * time / sample_rate) for time in
			 range(ceil(number_of_cycles * sample_rate / frequency))]
	if enable_hann:
		pulse = hann_window(pulse)
	if fill_zeros_to:
		if len(pulse) < fill_zeros_to:
			pulse = pulse + [0 for _ in range(fill_zeros_to - len(pulse))]
		else:
			raise ValueError("Pulse exceeds number of samples given by 'fill_zeros_to'")
	return pulse


def open_socket(hostname):
	if hostname.startswith('SN208'):
		hostname = bone_connect.get_ipv6_link_local_address_from_serial(hostname)

	return bone_connect(hostname)


def authenticate_socket(socket: bone_connect):
	questions = [
		inquirer.Text('username', message="Enter Username", default="admin"),
		inquirer.Password('password', message="Enter Password", default="")
	]

	answers = inquirer.prompt(questions)

	if answers is None:
		raise Exception("Login cancelled by user")

	username = answers.get("username")
	password = answers.get("password")		

	if username is None or len(username) == 0:
		raise Exception("Username cannot be empty")

	if password is None or len(password) == 0:
		raise Exception("Password cannot be empty")

	socket.login(username, password)


def plotFrequencyResponse(frequencies, gate_energies, title=None, ref=None, error=None):
	ax = plt.axes()
	ax.set_title(f"Frequency Response {title}")
	
	formatter_time = EngFormatter(unit="s")

	for i, (low, high) in enumerate(integral_gates):
		range_min = low / ADC_SAMPLE_RATE if low is not None else 0
		range_max = high / ADC_SAMPLE_RATE if high is not None else dv_length / ADC_SAMPLE_RATE
		
		label = f'Integrator Gate {i} ({formatter_time(range_min)} - {formatter_time(range_max)})' if i > 0 else f'Full Range ({formatter_time(range_min)} - {formatter_time(range_max)})'
		ax.scatter(frequencies, gate_energies[i], label=label, alpha=0.7, zorder=5-i)

	if ref:
		ax.scatter(ref[0], ref[1], color='gray', label='ref', alpha=0.1)

	f_min_energy = gate_energies[0][frequencies.index(f_min)]
	f_max_energy = gate_energies[0][frequencies.index(f_max)]

	relevant_slice = gate_energies[0][frequencies.index(f_min):frequencies.index(f_max)]

	max_energy = max(gate_energies[0])
	max_freq = frequencies[gate_energies[0].index(max_energy)]

	energy_min = min([f_min_energy, f_max_energy, max_energy])
	energy_max = max([f_min_energy, f_max_energy, max_energy])
	ripple = energy_max - energy_min
	att = sum(relevant_slice) / len(relevant_slice)

	graph_min_y = min([min(ge) for ge in gate_energies])
	graph_max_y = max([max(ge) for ge in gate_energies])
	interval = graph_max_y - graph_min_y
	margin = interval * 0.1

	graph_max_y = graph_max_y + margin
	graph_min_y = graph_min_y - margin

	err_text = ""
	text_color = 'gray'

	if error:
		err_text = f"\nerror: {error:.2f} dB"

		if error > 0.5:
			text_color = 'red'

	ax.text(0.985, 0.985, f"ripple: {ripple:.0f} dB\nattenuation: {att:.0f} dB{err_text}", verticalalignment='top', horizontalalignment='right', fontsize=24, color=text_color, transform=ax.transAxes)
	ax.text(0.015, 0.015, f"gain: {scaleVGA(1., vga):.0f}\nlevel: {level / 255. * 100.:.0f}%", verticalalignment='bottom', horizontalalignment='left', fontsize=24, color='gray', transform=ax.transAxes)

	ax.axhline(y=att, color='gray', linestyle='--', alpha=0.5)

	ax.axvline(x=f_min, color='gray', linestyle='--')
	ax.hlines([f_min_energy], [graph_min_y], [f_min], color='gray', linestyle='--', alpha=0.5)

	ax.axvline(x=f_max, color='gray', linestyle='--')
	ax.hlines([f_max_energy], [graph_min_y], [f_max], color='gray', linestyle='--', alpha=0.5)

	ax.vlines([max_freq], [graph_min_y], [max_energy], color='gray', linestyle='--', alpha=0.5)
	ax.axhline(y=max_energy, color='gray', linestyle='--')

	log = True

	if log:
		ax.text(41_000, att, f"{att:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
		ax.text(41_000, f_min_energy, f"{f_min_energy:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
		ax.text(41_000, max_energy, f"{max_energy:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
		ax.text(41_000, f_max_energy, f"{f_max_energy:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
	else:
		ax.text(5_000, att, f"{att:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
		ax.text(5_000, f_min_energy, f"{f_min_energy:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
		ax.text(5_000, max_energy, f"{max_energy:.0f} dB", verticalalignment='bottom', horizontalalignment='left')
		ax.text(5_000, f_max_energy, f"{f_max_energy:.0f} dB", verticalalignment='bottom', horizontalalignment='left')

	formatter_energy = EngFormatter(unit="dB")
	formatter_frequency = EngFormatter(unit="Hz")
	ax.yaxis.set_major_formatter(formatter_energy)

	ax.set_ylim(graph_min_y, graph_max_y)
	
	if log:
		ax.set_xlim(40_000, 1_200_000)
		ax.set_xscale('log')
	else:
		ax.set_xlim(0, 1_200_000)

	ax.xaxis.set_major_formatter(formatter_frequency)
	ax.set_xticks([100_000, max_freq , 1_000_000])

	ax.grid(True)
	ax.grid(which='major', linestyle=':', alpha=0.6)
	ax.grid(which='minor', linestyle=':', alpha=0.3)

	ax.legend()

	plt.show()


def getVGA(socket):
	data = socket.send_message({'command': 'channel_attributes', 'api': 2, 'payload': {'name': 'global'}})
	return data["payload"]["global"]["vga"]


def scaleVGA(input, vga):
	if vga == 0:
		vga_scaler = 0.5
	else:
		vga_scaler = float(1 << vga - 1)
	
	return input / vga_scaler


def calculateIntegral(dv: list, vga: int, low: Optional[int] = None, high: Optional[int] = None):
	if low is None:
		low = 0

	if high is None:
		high = len(dv) - 1
		
	gate_energy = sum([abs(x) for x in dv[low:high]]) * (1. / ADC_SAMPLE_RATE)
	return scaleVGA(gate_energy, vga)


def calculateMax(dv: list, vga: int, low: Optional[int] = None, high: Optional[int] = None):
	if low is None:
		low = 0

	if high is None:
		high = len(dv) - 1
		
	gate_energy = max([abs(x) for x in dv[low:high]])
	return scaleVGA(gate_energy, vga)


def calculateDvEnergy(dv, vga, use_integral_measurement=False):
	gate_energies = []
	for (low, high) in integral_gates:
		if use_integral_measurement:
			gate_energy = calculateIntegral(dv, vga, low, high)
		else:
			gate_energy = calculateMax(dv, vga, low, high)
		
		gate_energies.append(gate_energy)

	return gate_energies


def genDv(frequency):
	pulse = generate_sin_pulse(number_of_cycles = 5, frequency = frequency, 
			enable_hann = True, sample_rate = DAC_SAMPLE_RATE)
	pulse = pulse[:1023]
	return [round((sample + 1.) * 127.) for sample in pulse]


def setFreqGetEnergy(bone, frequency, avg=1, use_integral_measurement=False):
	pulse = genDv(frequency)
	bone.send_message({'command':'arbitrary', 'payload': {'len':len(pulse), 'arbitrary_data':pulse}})

	gate_energies = [0.0 for _ in integral_gates]

	for _ in range(avg):
		time.sleep(0.1)
		dv = bone.dv_data()

		global dv_length
		dv_length = len(dv)

		gate_energies_temp = calculateDvEnergy(dv, getVGA(bone), use_integral_measurement)

		for i, e in enumerate(gate_energies_temp):
			gate_energies[i] = gate_energies[i] + e

	gate_energies_db = []

	for e in gate_energies:
		e = e / avg
		e_db = getCalibratedEnergy(frequency, e)
		e_db = e_db / (20. / 256. * level)
		e_db = 20 * log10(e_db)
		gate_energies_db.append(e_db)

	return gate_energies_db


def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument("hostname", help="Hostname or serial number of the BeMoS controller to connect to")
	parser.add_argument("--high_res", action="store_true", help="Use high resolution frequency steps (1 kHz instead of 10 kHz)")
	parser.add_argument("--avg", nargs="?", default=1, type=int, help="Number of measurements to average per frequency (default: 1)")
	parser.add_argument("--ref", nargs="?", type=str, help="Reference CSV file for comparison")
	parser.add_argument("--out", nargs="?", type=str, help="Output CSV file to save frequency response")
	parser.add_argument("--calib", nargs="?", action="append", type=str, help="Calibration CSV file")
	parser.add_argument("--fmin", nargs="?", type=int, default=f_min, help=f"Minimum sweep frequency (default: {f_min} Hz)")
	parser.add_argument("--fmax", nargs="?", type=int, default=f_max, help=f"Maximum sweep frequency (default: {f_max} Hz)")
	parser.add_argument("--level", nargs="?", type=float, default=level, help=f"Output level (0-255) (default: {level:.0f})")
	parser.add_argument("--vga", nargs="?", type=int, default=vga, help=f"VGA setting (0-7) (default: {vga})")
	parser.add_argument("--gates", action="store_true", help="Enable integrator gate measurement")
	parser.add_argument("--use_integral_measurement", action="store_true", help="Use integral measurement instead of peak measurement")
	return parser.parse_args()


def getRefData(file):
	df = pd.read_csv(file)
	return df['Frequency [Hz]'].tolist(), df['Amplitude [dB]'].tolist()


def saveData(file, freqs, gate_energies):
	data = {'Frequency [Hz]': freqs}
	for i, gate_energy_list in enumerate(gate_energies):
		if i == 0:
			data['Amplitude [dB]'] = gate_energy_list
		else:
			data[f'Integrator Gate {i+1} [dB]'] = gate_energy_list
	
	df = pd.DataFrame(data)
	df.to_csv(file, index=False, lineterminator='\n')


calib_functions = []
combined_function = None

def loadCalibration(file):
	global calib_functions, combined_function
	df = pd.read_csv(file, skipinitialspace=True)

	calib_freqs = df['Frequency [Hz]'].tolist()
	energies = df['Amplitude [dB]'].tolist()
	calib_gains = [10 ** (energy / 20) for energy in energies]

	temp_function = interp1d(
		calib_freqs, 
		calib_gains, 
		kind='cubic',
		bounds_error=True
	)

	calib_functions.append(temp_function)
	combined_function = lambda f: np.prod([func(f) for func in calib_functions])

def getCalibratedEnergy(frequency, energy):
	if combined_function is None:
		return energy
	
	return energy / combined_function(frequency)

def interpolateFromList(x, x_list, y_list):
	return np.interp(x, x_list, y_list)


def calculateRefError(freqs, energies, ref):
	ref_energies = [interpolateFromList(freq, ref[0], ref[1]) for freq in freqs]
	error = [energy - ref_energy for energy, ref_energy in zip(energies, ref_energies)]
	return error

def pullIntegratorGates(bone: bone_connect):
	global integral_gates
	response = bone.send_message(
		{'command': 'channel_attributes', 'payload': {'name': 'global'}}
	)

	try:
		gate_0_low = response["payload"]["global"]["integrator_gate_low"]
		gate_0_high = response["payload"]["global"]["integrator_gate_high"]
		integral_gates.append((gate_0_low, gate_0_high))

		gate_1_low = response["payload"]["global"]["integrator_1_gate_low"]
		gate_1_high = response["payload"]["global"]["integrator_1_gate_high"]
		integral_gates.append((gate_1_low, gate_1_high))
	except KeyError as e:
		print(f"Warning: Could not retrieve integrator gate settings from device. ({e})")
		integral_gates = []


def main():
	global f_min, f_max

	args = parseArgs()

	if args.fmin:
		f_min = args.fmin

	if args.fmax:
		f_max = args.fmax

	freqs = []
	gate_energies = []
	ref = [[], []]

	if args.calib:
		for calib_file_path in args.calib:
			with open(calib_file_path, 'r') as calib_file:
				loadCalibration(calib_file)

	if args.ref:
		with open(args.ref, 'r') as ref_file:
			ref[0], ref[1] = getRefData(ref_file)

	with open_socket(args.hostname) as bone:
		authenticate_socket(bone)
		bone.send_message({'command':'stimulus', 'payload': {'mode':6}})
		bone.send_message({'command':'channel_attributes', 'payload': {'name': 'autoconfig', 'data': {'autolevel': False, 'autovga': False}}})
		bone.send_message({'command':'vga', 'payload': {'vga': args.vga}})
		bone.send_message({'command':'level', 'payload': {'level': args.level}})

		if args.gates:
			pullIntegratorGates(bone)

		for frequency in tqdm(range(50_000, 1_100_001, 1_000 if args.high_res else 10_000)):
			temp_gate_energies = setFreqGetEnergy(bone, frequency, args.avg, args.use_integral_measurement)

			freqs.append(frequency)

			if len(gate_energies) == 0:
				gate_energies = [[] for _ in integral_gates]
			
			for i, e in enumerate(temp_gate_energies):
				gate_energies[i].append(e)

		serial = bone.send_message({'command':'serial_number'})["payload"]["serial_number"]

		if args.out:
			with open(args.out, 'w') as out_file:
				saveData(out_file, freqs, gate_energies)

		error = None

		if len(ref[0]) > 0 and len(ref[1]) > 0:
			idx_min = freqs.index(f_min)
			idx_max = freqs.index(f_max)
			relevant_frequencies = freqs[idx_min:idx_max]
			relevant_energies = gate_energies[0][idx_min:idx_max]
			error = calculateRefError(relevant_frequencies, relevant_energies, ref)
			error = sum([abs(e) for e in error]) / len(error)

		plotFrequencyResponse(freqs, gate_energies, serial, ref, error)


if __name__ == "__main__":
	main()
