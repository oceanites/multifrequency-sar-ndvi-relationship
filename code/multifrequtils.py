from datetime import date
from functools import partial
import pandas as pd
import numpy as np
import rasterio
from rasterstats import zonal_stats
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
from rasterio.io import MemoryFile
import zipfile
import scipy.ndimage


def add_cr(
        data: pd.DataFrame,
        x="median_novasar",
        cross_pol="HV",
        co_pol="HH",
        date_col="date_novasar",
        cr_col="CR",
):
    data = data.copy()
    # data.rename(columns={x: "value"}, inplace=True)

    # Transform dataframe to have (id, date): [HH, HV] shape
    pivot_df = data.pivot(index=('polygon_id', date_col),
                          columns='polarization',
                          values=x)
    # actual CR calculation: CR = pol1 / pol2
    # in log scale the division is transformed into a subtraction
    pivot_df[cr_col] = pivot_df[co_pol] - pivot_df[cross_pol]
    pivot_df = pivot_df.reset_index()

    # copy metadata into CR column
    metadata_columns = set(data.columns) - set(["polygon_id", date_col, x])
    metadata_columns = list(metadata_columns)
    metadata_df = data.drop_duplicates(subset=('polygon_id', date_col))[['polygon_id', date_col] + metadata_columns]
    pivot_df = pd.merge(pivot_df, metadata_df, on=('polygon_id', date_col), how='left')

    ratio_df = pivot_df[['polygon_id', cr_col, date_col] + metadata_columns]
    ratio_df.loc[:, 'polarization'] = cr_col
    ratio_df = ratio_df.rename(columns={cr_col: x})

    data = data.reset_index(drop=True)
    ratio_df = ratio_df.reset_index(drop=True)

    # pd.concat() throws "InvalidIndexError: Reindexing only valid with uniquely valued Index objects"
    # therefore we transform the dataframe in a list({column1: value, column2: value, ...}) object
    # concat these and transform back to a dataframe
    # result_df = pd.concat([data, ratio_df], ignore_index=True)
    data_list = data.to_dict(orient="records")
    ratio_list = ratio_df.to_dict(orient="records")
    result_df = pd.DataFrame(data_list + ratio_list)
    return result_df


def read_s2_zip_scl(s2_zip_path):
    with zipfile.ZipFile(s2_zip_path, 'r') as z:
        scl_paths = list(filter(lambda f: 'SCL_20m.jp2' in f, z.namelist()))
        assert len(scl_paths) == 1
        scl_path = scl_paths[0]

        with z.open(scl_path) as f:
            with MemoryFile(f.read()) as memfile:
                with memfile.open() as src:
                    # cloud_probabilities_20m = src.read(1)
                    scl_20m = src.read(1)
    return scl_20m


def resample_20m_to_10m(raster_20m, spline_order=0):
    raster_10m = scipy.ndimage.zoom(input=raster_20m, zoom=2, order=spline_order)
    return raster_10m


def open_s2_zip(s2_zip_path, bands=None, cloud_mask_threshold: int = 60):
    band_resolution = '10m'
    if not bands:
        bands = ['B04', 'B08']

    band_arrays = list()
    profile = None
    cloud_mask = None
    mask_path = None

    with zipfile.ZipFile(s2_zip_path, 'r') as z:
        # extract raster band names
        band_paths = list()
        for band in bands:
            _band_paths = list(filter(lambda f: f'{band}_{band_resolution}.jp2' in f, z.namelist()))
            assert len(_band_paths) == 1, (f'{band}_{band_resolution}.jp2 not found in zip')
            band_paths.extend(_band_paths)

        # Read the raster bands
        for band_path in band_paths:
            with z.open(band_path) as f:
                with MemoryFile(f.read()) as memfile:
                    with memfile.open() as src:
                        band_data = src.read(1)
                        band_arrays.append(band_data)

                        # Copy the profile for the first band and modify it for multi-band
                        if profile is None:
                            profile = src.profile
    # modify the profile for multi-band
    profile.update(count=len(band_arrays), nodata=np.nan)

    # Create an in-memory multi-band raster dataset
    multi_band_array = np.stack(band_arrays)
    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(multi_band_array)

    return memfile


def open_s2_and_save_ndvi_img(s2_zip_path, ndvi_tif_dir):
    ndvi_tif_path = s2_zip_path.parent / (s2_zip_path.stem + "_NDVI" + ".tif")
    if ndvi_tif_path.is_file():
        print("Skipping existing tif.", ndvi_tif_path)
        return

    bands = ['B04', 'B08']
    cloud_mask_threshold = 40
    scl_mask_values = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]
    # Exclude: No Data, Saturated or defective pixel, Topographic casted shadows, Water, Unclassified, 
    # Cloud shadows, Cloud medium|high probability, cirrus, ice/snow
    s2_memfile = open_s2_zip(s2_zip_path=s2_zip_path, bands=bands)
    with s2_memfile.open() as s2_zip:
        s2_raster = s2_zip.read().astype(np.float32)
        s2_raster_profile = s2_zip.profile
    assert s2_raster.shape[0] == len(bands)

    ### NDVI calculation
    s2_raster = np.clip(s2_raster, 1_000,
                        11_000) - 1000  # to have reflectance from 0-10k, clipping to avoid integer underflow
    ndvi_raster = (s2_raster[1] - s2_raster[0]) / (s2_raster[1] + s2_raster[0] + 1e-9)
    ndvi_raster = np.clip(ndvi_raster, -1, 1)  # this should already be bound to this, but is not, so do it manually
    scl_raster_10m = resample_20m_to_10m(read_s2_zip_scl(s2_zip_path))

    mask = (scl_raster_10m == scl_mask_values[0])
    for scl_mask_value in scl_mask_values:
        mask |= (scl_raster_10m == scl_mask_value)

    ndvi_masked = np.where(mask, np.nan, ndvi_raster)
    ndvi_masked = ndvi_masked.astype(np.float32)

    with rasterio.open(ndvi_tif_path, "w", driver='GTiff',
                       height=s2_raster_profile["height"],
                       width=s2_raster_profile["width"],
                       dtype=ndvi_masked.dtype,
                       crs=s2_raster_profile["crs"],
                       transform=s2_raster_profile["transform"],
                       count=1,
                       compress="PACKBITS",
                       ) as ndvi_ds:
        ndvi_ds.write(ndvi_masked, 1)


def compute_zonal_stats_rasterio(raster_path, polygons, band_idx=0):
    with rasterio.open(raster_path) as src:
        stats_list = compute_zonal_stats_rasterio_opened(polygons=polygons, raster_ds=src, band_idx=band_idx)
    stats_list = [s for s in stats_list if s["count"] > 10]
    return stats_list


def compute_zonal_stats(raster_path, polygons):
    with rasterio.open(raster_path) as ds:
        raster_crs = ds.crs
    assert polygons.crs == raster_crs, ("CRS mismatch!", polygons.crs, raster_crs)
    stats = zonal_stats(polygons, raster_path, stats=['median', "count"])  # 'std', 'min', 'max', 'mean'])
    return stats


def compute_zonal_stats_rasterio_opened(polygons: gpd.GeoDataFrame, raster_ds, raster_transformation_func=None,
                                        band_idx=0):
    raster_crs = raster_ds.crs
    if raster_crs != polygons.crs:
        polygons = polygons.to_crs(raster_crs)

    stats_list = []

    for _, geom in polygons.iterrows():
        try:
            out_image, out_transform = rasterio.mask.mask(raster_ds,
                                                          [geom['geometry']],
                                                          crop=True,
                                                          filled=False,  # returns masked array
                                                          )
            if raster_transformation_func:
                out_image = raster_transformation_func(out_image)

            # also mask NaN values
            out_image.mask |= (~np.isfinite(out_image))
            out_image = out_image[band_idx]  # Assuming single band

            # Calculate statistics
            # masked_data = np.ma.masked_array(out_image, mask=(out_image == raster_ds.nodata))
            masked_data = out_image
            mean = np.nanmean(masked_data)
            median = np.ma.median(masked_data)
            count = np.ma.count(masked_data)
        except ValueError as e:
            # Handle case where polygon does not overlap with raster
            mean = np.nan
            count = np.nan
            median = np.nan

        stats = {
            'mean': mean,
            'median': median,
            "count": count,
        }
        if "polygon_id" in geom:
            stats["polygon_id"] = geom["polygon_id"]

        stats_list.append(stats)
    return stats_list


def get_today_str() -> str:
    """Return the iso-format string of todays date"""
    today_str = date.today().isoformat()
    return today_str


def get_top_two_classes_statistics(row: pd.Series, class_columns) -> pd.Series:
    """Get the most frequent class and if there is a second class having >30% of the pixels, also this."""
    data = pd.Series(row[class_columns]).astype(float)
    top_two = data.nlargest(2)  # Get two largest values (pixel counts)
    total_pixels = row['pixel_sum']
    if total_pixels <= 0:
        print("No pixels for polygon", row["polygon_id"])
        dominant_class_name, dominant_class_fraction = None, None
        side_class_name, side_class_fraction = None, None
    else:
        dominant_class_name = top_two.index[0]  # The class with the highest pixels
        dominant_class_fraction = row[dominant_class_name] / total_pixels
        side_class_name = top_two.index[1]
        side_class_fraction = row[side_class_name] / total_pixels
        if np.isnan(side_class_fraction):
            side_class_name = None
    out = [dominant_class_name, dominant_class_fraction, side_class_name, side_class_fraction]
    out = pd.Series(out)
    return out


def compute_zonal_classification_stats(classifications_raster, polygons, category_map: dict = None) -> pd.DataFrame:
    with rasterio.open(classifications_raster) as ds:
        assert ds.count == 1, ("Expected only one band!", ds.count)

        raster_crs = ds.crs
        classifications = ds.read()
    assert polygons.crs == raster_crs, ("CRS mismatch!", polygons.crs, raster_crs)
    classifications = np.ma.masked_invalid(classifications)
    int_classes = np.ma.unique(classifications)
    no_classes = int_classes.count()  # int_classes is a masked array for which count() returns the number of non-masked elements

    stats = zonal_stats(polygons, classifications_raster,
                        categorical=True, nodata=np.nan, geojson_out=True,
                        category_map=category_map)
    stats_df = gpd.GeoDataFrame.from_features(stats)
    class_columns = set(stats_df.columns) & set(category_map.values())  # & is intersection of sets
    class_columns = list(class_columns)
    get_filtered_pearsons_r
    stats_df.loc[:, "pixel_sum"] = stats_df.loc[:, class_columns].sum(axis="columns")
    get_top_two_classes_colnames = partial(get_top_two_classes_statistics, class_columns=class_columns)
    class_stat_cols = ['primary_class', 'primary_class_fraction', 'secondary_class', 'secondary_class_fraction']
    stats_df[class_stat_cols] = stats_df.apply(get_top_two_classes_colnames, axis=1)

    return stats_df


def get_s2_zip_files(s2_img_dir):
    zip_files = list()
    for z in s2_img_dir.glob("*.zip"):
        assert z.is_file(), z
        zip_files.append(z)
    return zip_files


def get_s2_ndvi_files(ndvi_img_dir):
    ndvi_files = list(ndvi_img_dir.glob("*_NDVI.tif"))
    return ndvi_files


def get_filtered_pearsons_r(x, y, data, outlier_value=0.01) -> dict:
    filt = data.copy()
    if "mndwi_mean" in filt.columns:
        filt = filt[filt.mndwi_mean <= 0.2]
        print("Filtered water pixels away")
    x_min, x_max = filt[x].quantile([outlier_value, 1 - outlier_value])
    y_min, y_max = filt[y].quantile([outlier_value, 1 - outlier_value])
    filt = filt[filt[x] > x_min]
    filt = filt[filt[x] < x_max]
    filt = filt[filt[y] > y_min]
    filt = filt[filt[y] < y_max]

    result = scipy.stats.pearsonr(filt[x], filt[y])
    if result.pvalue > 0.001:
        print(f"High p value! {result.pvalue:.3f}")
    return {"r": result.statistic, "p": result.pvalue, "N": len(filt)}


def scatterplot_nice(x, y, data, xlabel, ylabel,
                     hue=None, huelabel=None, xlim=None, title=None, ylim=(0.1, 1), alpha=1., style=None):
    sns.set_theme(style="whitegrid")
    cmap = sns.color_palette('crest', as_cmap=True)
    if not hue:
        scatter = sns.scatterplot(x=x, y=y, hue=hue, data=data, legend=False, alpha=alpha, style=style)
    else:
        # categorical data
        if data[hue].dtype == np.dtype("O"):
            scatter = sns.scatterplot(x=x, y=y, hue=hue, data=data, alpha=alpha, style=style)
            sns.move_legend(scatter, loc="best", title=huelabel)

        else:  # numerical data
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
            scatter = sns.scatterplot(x=x, y=y, hue=hue, data=data, style=style,
                                      hue_norm=sm.norm, palette=cmap, legend=False,
                                      alpha=alpha)
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label(huelabel)
    scatter.set_xlim(xlim)
    scatter.set_ylim(ylim)
    scatter.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.tight_layout()
