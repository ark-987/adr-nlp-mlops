import os
import zipfile
import shutil
from google.cloud import storage

def get_gcs_client():
    """Attempts real cloud authentication with explicit private key routing configurations."""
    # Look for the private key path mapped by your docker-compose environment variables
    explicit_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    try:
        if explicit_key_path and os.path.exists(explicit_key_path):
            # Authenticate securely using your secrets/gcp-key.json file
            return storage.Client.from_service_account_json(explicit_key_path)
        
        # Fallback to standard implicit cloud paths if running directly on GCP
        return storage.Client(project="project-edc577ab-2a6a-4016-89e")
    except Exception as e:
        print(f"[INFO] Google authentication unavailable ({e}). Activating Local Storage Fallback Mode.")
        return None


# =========================================================================
# NEW CRITICAL FUNCTION: Production Model Downloader & Unpacker
# =========================================================================
def download_and_extract_model_from_gcs(bucket_name, source_blob_name, extract_to_dir="models/adr-nlp-final"):
    """Downloads the fine-tuned BioBERT zip bundle from GCS and extracts shards to container memory."""
    client = get_gcs_client()
    os.makedirs(extract_to_dir, exist_ok=True)
    local_zip_path = os.path.join("models", "temp_model.zip")

    # REAL CLOUD DOWNLOAD OPERATION
    if client is not None:
        try:
            print(f"[BOOT] Reaching out to cloud bucket: gs://{bucket_name}...")
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            
            # Pull down the model zip archive package over network channels
            blob.download_to_filename(local_zip_path)
            print(f"  Downloaded model archive package successfully to {local_zip_path}")
            
            # Extract the raw Hugging Face shards directly into target storage directories
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to_dir)
            
            # Clean up the raw zip file to optimize container volume usage
            os.remove(local_zip_path)
            print(f"  BioBERT weights unpacked cleanly inside: {extract_to_dir}/")
            return True
        except Exception as e:
            print(f"[CLOUD ERROR] Model download sequence failed: {e}. Attempting local mock fallback...")

    # OFFLINE BACKUP SIMULATION (For local developer convenience)
    print(f"[MOCK GCS DOWNLOAD] Searching for mock weights inside data/mock_gcs_bucket/...")
    mock_source_dir = os.path.join("data", "mock_gcs_bucket", os.path.dirname(source_blob_name))
    
    if os.path.exists(mock_source_dir):
        shutil.copytree(mock_source_dir, extract_to_dir, dirs_exist_ok=True)
        print(f"  Mock BioBERT shards cloned successfully from workspace folder cache into {extract_to_dir}")
        return True
    
    print(f"[CRITICAL FAILURE] No model assets found in cloud or local mock caches!")
    return False


# =========================================================================
# EXISTING UPLOAD PIPELINE FUNCTIONS
# =========================================================================
def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    client = get_gcs_client()
    
    if client is not None:
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to gs://{bucket_name}/{destination_blob_name}")
            return True
        except Exception as e:
            print(f"[CLOUD ERROR] Upload failed: {e}. Falling back to local replication...")

    print(f"[MOCK GCS UPLOAD] Re-routing file to simulated local workspace storage...")
    mock_cloud_dir = os.path.join("data", "mock_gcs_bucket", os.path.dirname(destination_blob_name))
    os.makedirs(mock_cloud_dir, exist_ok=True)
    shutil.copy(local_file_path, os.path.join(mock_cloud_dir, os.path.basename(destination_blob_name)))
    print("  Asset safely synced inside mock cloud folder structure!")
    return True

def upload_directory_to_gcs(local_dir, bucket_name, destination_gcs_dir):
    client = get_gcs_client()
    
    if client is not None:
        try:
            bucket = client.bucket(bucket_name)
            for root, _, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, local_dir)
                    blob_path = os.path.join(destination_gcs_dir, relative_path).replace("\\", "/")
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_path)
            print(f"Uploaded directory {local_dir} to gs://{bucket_name}/{destination_gcs_dir}")
            return True
        except Exception as e:
            print(f"[CLOUD ERROR] Directory sync failed: {e}. Falling back to local replication...")

    print(f"[MOCK GCS UPLOAD] Re-routing directory splits to simulated local workspace storage...")
    mock_cloud_dir = os.path.join("data", "mock_gcs_bucket", destination_gcs_dir)
    os.makedirs(mock_cloud_dir, exist_ok=True)
    if os.path.exists(local_dir):
        shutil.copytree(local_dir, mock_cloud_dir, dirs_exist_ok=True)
        print("  Directory splits safely synced inside mock cloud folder structure!")
    return True
