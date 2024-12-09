"""
Script to process SAR data and compute the Dual-Polarization Radar Vegetation Index (DpRVI).

This script supports either:
- A single raster file with two bands (representing two polarizations) or
- Two raster files, each containing one band corresponding to a specific polarization.

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

def compute_dprvi(co_pol, cross_pol, is_log_scaled: bool):
    """
    Compute the Dual-polarization Radar Vegetation Index (DpRVI).

    This is defined as '(4*cross) / (cross + co)' with cross/co the cross- and co-
    polarized backscatter values in linear scale.
    
    Arguments:
    - co_pol: numpy array representing the first polarization (e.g., VV, HH).
    - cross_pol: numpy array representing the second polarization (e.g., VH, HV).
    - is_log_scaled: whether the data is in log scale.
    
    Returns:
    - DpRVI: numpy array containing the computed DpRVI values.
    """
    if is_log_scaled:
        co_pol = 10 ** (co_pol / 10.0)
        cross_pol = 10 ** (cross_pol / 10.0)

    total_power = co_pol + cross_pol
    dprvi = np.divide(4 * cross_pol, total_power, where=total_power != 0)
    return dprvi

def process_input_files(input_files: list, polarizations: list):
    """
    Process the input files and return the bands as numpy arrays.
    
    Arguments:
    - input_files: list of 1 or 2 input file paths.
    - polarizations: tuple or list containing the polarizations of the files or bands.
    
    Returns:
    - co_pol: numpy array of the co-polarized band.
    - cross_pol: numpy array of the cross-polarized band.
    """
    if len(input_files) == 1:
        with rasterio.open(input_files[0]) as src:
            co_pol = src.read(1) if (polarizations[0] in {'VV', 'HH'}) else src.read(2)
            cross_pol = src.read(2) if (polarizations[1] in {'VH', 'HV'}) else src.read(1)
    else:
        co_pol_file = input_files[0] if (polarizations[0] in {'VV', "HH"})  else input_files[1]
        cross_pol_file = input_files[1] if (polarizations[1] in {'VH', "HV"}) else input_files[0]
        with rasterio.open(co_pol_file) as src_co, rasterio.open(cross_pol_file) as src_cross:
            co_pol = src_co.read(1)
            cross_pol = src_cross.read(1)
    
    return co_pol, cross_pol
    

def get_arguments():
    parser = argparse.ArgumentParser(description="Compute DpRVI from SAR data.")
    
    # Positional arguments: input files and output file
    parser.add_argument("input_files", nargs='+', help="1 or 2 input GeoTIFF files")
    parser.add_argument("output_file", help="Output GeoTIFF file path")
    
    # Polarizations flag
    parser.add_argument(
        "--polarizations", 
        nargs=2, 
        required=True, 
        choices=["VV", "VH", "HH", "HV"], 
        help="Specify the polarizations of the input data. E.g., --polarizations VV VH"
    )
    parser.add_argument(
        "--scale", 
        choices=["linear", "log"], 
        help="Specify if input data is in linear or log scale (default: linear)"
    )
    
    args = parser.parse_args()

    if len(args.input_files) not in [1, 2]:
        parser.error("You must provide either 1 or 2 input files.")

    return args


def main():
    args = get_arguments()
    is_log_scaled = (args.scale == "log")
    
    co_pol, cross_pol = process_input_files(args.input_files, args.polarizations)
    
    dprvi = compute_dprvi(co_pol=co_pol, cross_pol=cross_pol, is_log_scaled=is_log_scaled)

    # Write the output to a new GeoTIFF
    with rasterio.open(args.input_files[0]) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)
        
        with rasterio.open(args.output_file, 'w', **profile) as dst:
            dst.write(dprvi.astype(rasterio.float32), 1)
    
    print(f"DpRVI written to {args.output_file}")

if __name__ == "__main__":
    main()

# -


