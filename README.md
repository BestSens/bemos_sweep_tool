# BeMoS Sweep
## Basic usage
```bash
python ./sweep.py 192.168.100.61
```

With calibration file:
```bash
python ./sweep.py --calib calib_file.csv 192.168.100.61
```

With serial number to IPv6 conversion:
```bash
python ./sweep.py --calib calib_file.csv SN208148
```

## Commandline options
```
usage: sweep.py [-h] [--high_res] [--avg [AVG]] [--ref [REF]] [--out [OUT]] [--calib [CALIB]] [--fmin [FMIN]]
                [--fmax [FMAX]] [--level [LEVEL]] [--vga [VGA]] [--gates] [--use_integral_measurement]
                hostname

positional arguments:
  hostname              Hostname or serial number of the BeMoS controller to connect to

options:
  -h, --help            show this help message and exit
  --high_res            Use high resolution frequency steps (1 kHz instead of 10 kHz)
  --avg [AVG]           Number of measurements to average per frequency (default: 1)
  --ref [REF]           Reference CSV file for comparison
  --out [OUT]           Output CSV file to save frequency response
  --calib [CALIB]       Calibration CSV file
  --fmin [FMIN]         Minimum sweep frequency (default: 100000 Hz)
  --fmax [FMAX]         Maximum sweep frequency (default: 1000000 Hz)
  --level [LEVEL]       Output level (0-255) (default: 127)
  --vga [VGA]           VGA setting (0-7) (default: 1)
  --gates               Enable integrator gate measurement
  --use_integral_measurement
                        Use integral measurement instead of peak measurement
```

## Create calibration curve
Measure with transmission adapter plug and same cable length as normal measurements will be done. If possible, calibration should be done with the same VGA and level settings as the actual measurements for better comparability.
```bash
python ./sweep.py --high_res --avg 10 --out calib_file.csv 192.168.100.61
```

