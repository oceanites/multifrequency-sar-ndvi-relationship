#!/usr/bin/env python
# Version: 0.2
"""
This script processes NovaSAR Synthetic Aperture Radar (SAR) data, calibrates the backscatter
into linear scale using a calibration constant from an accompanying metadata file, and 
computes zonal statistics for each polarization band (HH and HV). The calibrated backscatter
values are converted to dB scale for further analysis.

Usage:
    python process_novasar.py --raster <novasar_raster_file> --polygons <polygons_geojson>

Arguments:
    --raster: Path to the input NovaSAR raster file (GeoTIFF format).
    --polygons: Path to the GeoJSON file containing the polygons for zonal statistics.
"""

import rasterio
import os
from pathlib import Path
import numpy as np
import sys
import rasterio.mask
import argparse


def calibrate_novasar_raster(arr: np.ndarray, calibration_constant=5.02e5):
    """Calibrate SAR backscatter: (DN)^2 / CalibrationConstant"""
    calibrated = (arr.astype(np.float32)**2) / calibration_constant
    return calibrated
    
def read_calibration_constant(metadata_filepath) -> float:
    with open(metadata_filepath, "r") as fd:
        lines = fd.readlines()
        
        cal_const_line = list(filter(lambda l: "CalibrationConstant" in l, lines))
        assert len(cal_const_line) == 1, ("Multiple or no calibration constants found", len(cal_const_line))
        cal_const_line = cal_const_line[0].strip()  # remove whitespace
        cal_const = cal_const_line.split(">")[1].split("<")[0]
        cal_const = float(cal_const)

        cal_status_line = list(filter(lambda l: "CalibrationStatus" in l, lines))
        assert len(cal_status_line) == 1
        cal_status_line = cal_status_line[0]
        assert "CALIBRATED" not in cal_status_line[0], "Image not calibrated"
    return cal_const


def get_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process and calibrate NovaSAR SAR data")
    parser.add_argument('--input_file', type=Path, required=True, help="Path to the NovaSAR raster file (GeoTIFF)")    
    parser.add_argument('--output_file', type=Path, required=True, help="Output path")
    parser.add_argument('--bands', type=int, nargs="+", required=True, help="Which bands should be calibrated? Starts at 0.")
    args = parser.parse_args()
    assert args.input_file.name != args.output_file.name, ("Input and output filename should differ!")
    return args

def try_get_metdata_file(args):
    option1 = args.input_file.parent / (args.input_file.stem + ".metadata.xml")
    option2 = args.input_file.parent / "metadata.xml"
    if option1.is_file():
        metadata_file = option1
    elif option2.is_file():
        metadata_file = option2
    else:
        raise FileNotFoundError(f"Couldn't find metadata file at '{option1}' or '{option2}'.")
    return metadata_file
    

def main():
    args = get_arguments()

    metadata_file = try_get_metdata_file(args)
    calibration_constant = read_calibration_constant(metadata_file)
    assert calibration_constant == 5.02e5, ("Calibration constant differs from the default value.", calibration_constant)

    # Read DNs
    with rasterio.open(args.input_file) as src:
        dn = src.read()
        dn_profile = src.profile

    # Convert to linear amplitudes
    linear_amplitudes = dn.copy().astype(np.float32)
    linear_amplitudes[args.bands] = calibrate_novasar_raster(arr=dn[args.bands], calibration_constant=calibration_constant)
    linear_profile = dn_profile.copy()
    linear_profile.update(dtype=rasterio.float32)
    

    # Write back to file.
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_file, 'w', **linear_profile) as dst:
        dst.write(linear_amplitudes.astype(rasterio.float32))
    
    print(f"Wrote linear amplitudes to {args.output_file}")
    

if __name__ == "__main__":
    main()