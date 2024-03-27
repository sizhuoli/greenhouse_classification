import sys
import resource
from osgeo import gdal


def gdal_progress_callback(complete, message, data):
    """Callback function to show progress during GDAL operations such gdal.Warp() or gdal.Translate().

    Expects a tqdm progressbar in 'data', which is passed as the 'callback_data' argument of the GDAL method.
    'complete' is passed by the GDAL methods, as a float from 0 to 1
    """
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def raster_copy(output_fp, input_fp, mode="warp", resample=1, out_crs=None, bands=None, bounds=None, bounds_crs=None,
                multi_core=False, pbar=None, compress=False, cutline_fp=None, resample_alg=gdal.GRA_Bilinear):
    """ Copy a raster using GDAL Warp or GDAL Translate, with various options.

    The use of Warp or Translate can be chosen with 'mode' parameter. GDAL.Warp allows full multiprocessing,
    whereas GDAL.Translate allows the selection of only certain bands to copy.
    A specific window to copy can be specified with 'bounds' and 'bounds_crs' parameters.
    Optional resampling with bi-linear interpolation is done if passed in as 'resample'!=1.
    """

    # Common options
    base_options = dict(
        creationOptions=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER",
                         "NUM_THREADS=ALL_CPUS"],
        callback=gdal_progress_callback,
        callback_data=pbar
    )
    if compress:
        base_options["creationOptions"].append("COMPRESS=LZW")
    if resample != 1:
        # Get input pixel sizes
        raster = gdal.Open(input_fp)
        gt = raster.GetGeoTransform()
        x_res, y_res = gt[1], -gt[5]
        base_options["xRes"] = x_res / resample,
        base_options["yRes"] = y_res / resample,
        base_options["resampleAlg"] = resample_alg

    # Use GDAL Warp
    if mode.lower() == "warp":
        warp_options = dict(
            dstSRS=out_crs,
            cutlineDSName=cutline_fp,
            outputBounds=bounds,
            outputBoundsSRS=bounds_crs,
            multithread=multi_core,
            warpOptions=["NUM_THREADS=ALL_CPUS"] if multi_core else [],
            warpMemoryLimit=1000000000,  # processing chunk size. higher is not always better, around 1-4GB seems good
        )
        return gdal.Warp(output_fp, input_fp, **base_options, **warp_options)

    # Use GDAL Translate
    elif mode.lower() == "translate":
        translate_options = dict(
            bandList=bands,
            outputSRS=out_crs,
            projWin=[bounds[0], bounds[3], bounds[2], bounds[1]] if bounds is not None else None,
            projWinSRS=bounds_crs,
        )
        return gdal.Translate(output_fp, input_fp, **base_options, **translate_options)

    else:
        raise Exception("Invalid mode argument, supported modes are 'warp' or 'translate'.")


def get_driver_name(extension):
    """Get GDAL/OGR driver names from file extension"""
    if extension.lower().endswith("tif"):
        return "GTiff"
    elif extension.lower().endswith("jp2"):
        return "JP2OpenJPEG"
    elif extension.lower().endswith("shp"):
        return "ESRI Shapefile"
    elif extension.lower().endswith("gpkg"):
        return "GPKG"
    else:
        raise Exception(f"Unable to find driver for unsupported extension {extension}")


def memory_limit(percentage: float):
    """Set soft memory limit to a percentage of total available memory."""
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * percentage), -1))
    # print(f"Set memory limit to {int(percentage*100)}% : {get_memory() * percentage/1024/1024:.2f} GiB")


def get_memory():
    """Get available memory from linux system.

    NOTE: Including 'SwapFree:' also counts cache as available memory (so remove it to only count physical RAM).
    This can still cause OOM crashes with a memory-heavy single thread, as linux won't necessarily move it to cache...
    """
    with open('/proc/meminfo', 'r') as mem_info:
        free_memory = 0
        for line in mem_info:
            if str(line.split()[0]) in ('MemFree:', 'Buffers:', 'Cached:', 'SwapFree:'):
                free_memory += int(line.split()[1])
    return free_memory


def memory(percentage):
    """Decorator to limit memory of a python method to a percentage of available system memory"""
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 / 1024
                print('Available memory: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator
