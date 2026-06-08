import os
import zipfile
import shutil
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def get_s3_client():
    """Initializes the AWS S3 client using automatic environmental authentication."""
    try:
        # Boto3 automatically scans for AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY 
        # inside your running Docker or GitHub environment variables.
        return boto3.client("s3")
    except Exception as e:
        print(f"[INFO] AWS credentials missing or blocked ({e}). Activating Local Fallback Mode.")
        return None

def download_and_extract_model_from_s3(bucket_name, source_s3_key, extract_to_dir="models/adr-nlp-final"):
    """Downloads your fine-tuned BioBERT zip bundle from Amazon S3 and extracts it to disk."""
    client = get_s3_client()
    
    # CRITICAL SECURITY ANCHOR: Guarantee parent paths exist for extraction targets
    os.makedirs(extract_to_dir, exist_ok=True)
    
    # Secure parent execution directory for temporary file processing
    os.makedirs("models", exist_ok=True)
    local_zip_path = os.path.join("models", "temp_model.zip")

    # 1. REAL AWS CLOUD DOWNLOAD OPERATION
    if client is not None:
        try:
            print(f"[BOOT] Reaching out to AWS cloud bucket: s3://{bucket_name}/{source_s3_key}...")
            client.download_file(bucket_name, source_s3_key, local_zip_path)
            print(f"  Downloaded model archive package successfully to {local_zip_path}")
            
            # Extract the raw Hugging Face shards
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_dir)
            
            # Clean up the raw zip file to save container space
            os.remove(local_zip_path)
            print(f"  BioBERT weights unpacked cleanly inside: {extract_to_dir}/")
            return True
        except (NoCredentialsError, PartialCredentialsError):
            print("[AWS ERROR] Invalid or missing AWS Credentials. Attempting local mock fallback...")
        except Exception as e:
            print(f"[AWS ERROR] Model download sequence failed: {e}. Attempting local mock fallback...")

    # 2. OFFLINE BACKUP SIMULATION (Aligned exclusively with your AWS migration schema)
    mock_source_dir = os.path.join("data", "mock_aws_s3_bucket", "models")
    print(f"[MOCK S3 DOWNLOAD] Searching for local fallback weights inside: {mock_source_dir}")
    
    if os.path.exists(mock_source_dir):
        shutil.copytree(mock_source_dir, extract_to_dir, dirs_exist_ok=True)
        print(f"  Mock BioBERT shards cloned successfully from workspace folder cache into {extract_to_dir}")
        return True
    
    print(f"[CRITICAL FAILURE] No model assets found in AWS cloud or local mock caches!")
    return False