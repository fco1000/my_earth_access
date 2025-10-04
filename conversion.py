import os
import subprocess

input_dir = "data"
output_dir = "converted"
os.makedirs(output_dir, exist_ok=True)

failed = []

for fname in os.listdir(input_dir):
    if not fname.endswith(".hdf"):
        continue

    infile = os.path.join(input_dir, fname)
    outfile = os.path.join(output_dir, fname.replace(".hdf", "_ndvi.tif"))

    #  Skip files already converted
    if os.path.exists(outfile):
        print(f" Skipping (already converted): {fname}")
        continue

    cmd = [
        "gdal_translate",
        f'HDF4_EOS:EOS_GRID:"{infile}":MODIS_Grid_16DAY_250m_500m_VI:"250m 16 days NDVI"',
        outfile
    ]

    print(f"Converting: {fname}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f" Done: {fname}")
    except subprocess.CalledProcessError as e:
        print(f" Failed: {fname}")
        failed.append(fname)

# Log failures for later
if failed:
    with open("failed_conversions.txt", "a") as f:
        for name in failed:
            f.write(name + "\n")
    print("\n Conversion failed for:")
    print("\n".join(failed))
    
    print("number of failed is: ",len(failed))
else:
    print("\n All files converted successfully!")
    
