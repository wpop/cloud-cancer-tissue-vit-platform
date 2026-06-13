"""
Download PCam files safely using gdown.

This script downloads compressed .h5.gz files only.
It does not automatically decompress them.
"""

from pathlib import Path
import subprocess


PCAM_FILES = {
    "train_x": "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
    "train_y": "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
    "valid_x": "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
    "valid_y": "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
    # "valid_y": "1bH8ZRbhSVAhScTS0p9-ZzGnX91c_HT3u",
    "test_x": "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
    "test_y": "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
}


def main() -> None:
    output_dir = Path("/mnt/4086152D86152546/MedicalDatasets/PCam/pcam")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, file_id in PCAM_FILES.items():
        filename = f"camelyonpatch_level_2_split_{name}.h5.gz"
        output_path = output_dir / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Already exists: {output_path}")
            continue

        url = f"https://drive.google.com/uc?id={file_id}"

        print(f"Downloading {name}...")
        subprocess.run(
            ["gdown", "--continue", url, "-O", str(output_path)],
            check=True,
        )

    print("Download complete.")


if __name__ == "__main__":
    main()
