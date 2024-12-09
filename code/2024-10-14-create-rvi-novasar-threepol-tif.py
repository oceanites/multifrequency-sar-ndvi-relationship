"""
Script to process SAR data and compute the Dual-Polarization Radar Vegetation Index (DpRVI).

This script expects
- three raster files, each containing one band corresponding to a specific polarization.

The output will be a new raster file containing the DpRVI calculation based on the given inputs.

Arguments:
    - `input_files`: 
        - 1 or 2 GeoTIFF files as input.
        - If 1 file is provided, it must contain 2 bands, each representing one polarization.
        - If 2 files are provided, each file must contain 1 band, each representing one polarization.
    - `output_file`: Path to save the output GeoTIFF containing the DpRVI result.
    - `polarizations`: Specify the polarization of each band/file. 
        - Options: "VV", "VH", "HH", "HV".
        - If 1 input file is provided, specify which polarization corresponds to each band (e.g., `--polarizations VV VH`).
        - If 2 input files are provided, specify which polarization each file corresponds to (e.g., `--polarizations VV VH`).
    - `scale`: Specify whether the input data is provided in "linear" or "log" scale (default: "linear").
    
    Output:
        A single GeoTIFF file containing the DpRVI result, saved at the specified output path.

    Example Usage:
        python compute_dprvi.py input1.tif input2.tif output.tif --polarizations VV VH --scale linear
"""


# +
import argparse
import rasterio
import numpy as np
from pathlib import Path

def compute_rvi(vv, hh, hv, is_log_scaled: bool):
    """
    Compute the Radar Vegetation Index (RVI).

    This is defined as '(8*HV) / (HH+VV+2HV)' with backscatter values in linear scale.
    """
    if is_log_scaled:
        hh = 10 ** (hh / 10.0)
        vv = 10 ** (vv / 10.0)
        hv = 10 ** (hv / 10.0)

    total_power = hh + vv + 2 * hv
    rvi = np.divide(8 * hv, total_power, where=total_power != 0)
    return rvi
    

def read_threepol_input_files(input_files: list, polarizations: list):
    """
    Process the input files and return the bands as numpy arrays.
    """
    assert len(input_files) == 3
    assert polarizations == ["VV", "HH", "HV"], ("Unexpected polzations!", polarizations)
    vv_file = input_files[0]
    assert "_VV_" in vv_file.stem
    hh_file = input_files[1]
    assert "_HH_" in hh_file.stem
    hv_file = input_files[2]
    assert "_HV_" in hv_file.stem
    with rasterio.open(vv_file) as srcvv, rasterio.open(hh_file) as srchh, rasterio.open(hv_file) as srchv:
        vv = srcvv.read(1)
        hh = srchh.read(1)
        hv = srchv.read(1)
    print("Read input files")
    return vv, hh, hv


def dn_to_linear_scale(dn_arr):
    # equation from metadata, this is euqivalent to the instruction of one email
    # 10 * log10((DN  / 14125.3754)**2)
    # and
    # 20 * log10(DN) - 83
    # are equivalent
    # **2 to go from amplitudes to intensity values
    lin_arr = (dn_arr.astype(float)  / 14125.3754)**2
    return lin_arr


def get_arguments():
    parser = argparse.ArgumentParser(description="Compute RVI from threepol SAR data.")
    
    # Positional arguments: input files and output file
    parser.add_argument("input_files", nargs='+', help="3 input GeoTIFF files", type=Path)
    parser.add_argument("output_file", help="Output GeoTIFF file path")
    
    # Polarizations flag
    parser.add_argument(
        "--polarizations", 
        nargs=3, 
        required=True, 
        choices=["VV", "VH", "HH", "HV"], 
        help="Specify the polarizations of the input data. E.g., --polarizations VV HH HV"
    )
    parser.add_argument(
        "--scale", 
        choices=["linear", "log"],
        required=True, 
        help="Specify if input data is in linear or log scale (default: linear)"
    )
    
    args = parser.parse_args()

    if len(args.input_files) != 3:
        parser.error("You must provide 3 input files.")

    return args


def main():
    args = get_arguments()
    is_log_scaled = (args.scale == "log")
    if not is_log_scaled:
        print("WARNING: ARE YOU REALLY SURE YOU HAVE DATA IN LINEAR SCALE AND NOT ONLY DN WHICH NEED CALIBRATION FIRST???????")
    
    vv, hh, hv = read_threepol_input_files(args.input_files, args.polarizations)
    #vv, hh, hv = tuple(map(dn_to_linear_scale, [vv_dn, hh_dn, hv_dn]))
    rvi = compute_rvi(vv=vv, hh=hh, hv=hv, is_log_scaled=is_log_scaled)

    # Write the output to a new GeoTIFF
    with rasterio.open(args.input_files[0]) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        
        with rasterio.open(args.output_file, 'w', **profile) as dst:
            dst.write(rvi.astype(rasterio.float32), 1)
    
    print(f"RVI written to {args.output_file}")

if __name__ == "__main__":
    main()

# -


