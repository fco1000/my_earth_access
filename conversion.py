import os
import subprocess

input_dir = "data"
output_dir = "converted"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".hdf"):
        infile = os.path.join(input_dir, fname)
        outfile = os.path.join(output_dir, fname.replace(".hdf", "_ndvi.tif"))

        cmd = [
            "gdal_translate",
            f'HDF4_EOS:EOS_GRID:"{infile}":MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"',
            outfile
        ]


        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
