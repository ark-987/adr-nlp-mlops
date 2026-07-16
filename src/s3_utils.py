import os
import shutil
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import (
    NoCredentialsError,
    PartialCredentialsError,
    ClientError,
)


def get_s3_client():
    """
    Create an S3 client using credentials supplied through
    the environment (GitHub Actions, EC2 IAM Role, or AWS keys).
    """
    try:
        return boto3.client("s3")
    except Exception as e:
        print(f"[S3] Unable to initialise S3 client: {e}")
        return None


def download_and_extract_model_from_s3(
    bucket_name: str,
    source_s3_key: str,
    extract_to_dir: str,
):
    """
    Download the production BioClinicalBERT model from S3
    and extract it into the supplied directory.

    Returns
    -------
    bool
        True if a usable model is available.
    """

    client = get_s3_client()

    extract_path = Path(extract_to_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    temp_zip = extract_path.parent / "model.zip"

    # -------------------------------------------------------
    # Production path
    # -------------------------------------------------------

    if client is not None:

        try:

            print(
                f"[BOOT] Downloading model from "
                f"s3://{bucket_name}/{source_s3_key}"
            )

            client.download_file(
                bucket_name,
                source_s3_key,
                str(temp_zip),
            )

            print("[BOOT] Model archive downloaded.")

            with zipfile.ZipFile(temp_zip, "r") as archive:
                archive.extractall(extract_path)

            temp_zip.unlink(missing_ok=True)

            print(
                f"[BOOT] Model extracted to {extract_path}"
            )

            return True

        except (
            NoCredentialsError,
            PartialCredentialsError,
        ):

            print(
                "[AWS] Missing AWS credentials."
            )

        except ClientError as e:

            print(
                f"[AWS] S3 download failed: {e}"
            )

        except zipfile.BadZipFile:

            print(
                "[BOOT] Downloaded model archive is corrupted."
            )

        except Exception as e:

            print(
                f"[BOOT] Unexpected error: {e}"
            )

    # -------------------------------------------------------
    # Local testing fallback
    # -------------------------------------------------------

    mock_model = (
        Path("data")
        / "mock_aws_s3_bucket"
        / "models"
    )

    print(
        "[BOOT] Attempting local fallback model..."
    )

    if mock_model.exists():

        shutil.copytree(
            mock_model,
            extract_path,
            dirs_exist_ok=True,
        )

        print(
            "[BOOT] Local fallback model loaded."
        )

        return True

    print(
        "[BOOT] No model available from S3 or local fallback."
    )

    return False