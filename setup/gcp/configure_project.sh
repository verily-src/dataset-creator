#!/bin/bash

# =========================== Validate preconditions  ==========================

usage() {
  echo "Usage: $0 [-u] <project_id> <repo_access_token>"
  echo "  -u: An optional flag, indicating a configuration upgrade."
  echo "  project_id: The name of the GCP project to config dataset creator to."
  echo "  repo_access_token: A github access token with repo permissions."
  echo ""
  echo "Access tokens can be created by configuring in:"
  echo "Github.com > Settings > Developer settings > Personal access tokens."
  echo "Please make sure to only grant repo permissions to this access token."
  exit 1
}

do_upgrade=false
while getopts "u" upgrade; do
  case $upgrade in
    u)
      do_upgrade=true
      echo "Upgrading configuration."
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      usage
      ;;
  esac
done

# Shift the parsed options out of the argument list
shift $((OPTIND-1))

if ! [ "$(basename $(pwd))" = "oss-dataset-creator" ]; then
  echo "Please run this script from workdir oss-dataset-creator"
  exit 1
fi

# Check if the project ID argument is provided
if [ $# -ne 2 ]; then
  usage
fi

log () {
  echo "[**] $(date +%H:%M:%S.%N): $1"
}

# ============================= Install gCloud CLI =============================

# Extract the project ID from the command line argument
project_id="$1"
repo_access_token="$2"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    log "Google Cloud SDK is not installed. Installing..."

    # Download and install the Google Cloud SDK
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-444.0.0-linux-x86_64.tar.gz
    tar -xvf google-cloud-cli-444.0.0-linux-x86_64.tar.gz
    chmod +x ./google-cloud-sdk/install.sh
    ./google-cloud-sdk/install.sh

    # Check installation status
    if [ $? -eq 0 ]; then
        log "Google Cloud SDK has been successfully installed."
        gcloud auth login
    else
        log "Failed to install Google Cloud SDK."
    fi
else
    log "Google Cloud SDK is already installed."
fi

# =================== Verify necessary roles configuration =====================

log "Validating necessary roles configuration..."

project_number=$(
    gcloud projects describe "$project_id" --format="value(projectNumber)"
)
default_compute_account="$project_number-compute@developer.gserviceaccount.com"
default_cloudbuild_account="$project_number@cloudbuild.gserviceaccount.com"

print_role_if_missing () {
  email="$1"
  role="$2"

  if ! gcloud projects get-iam-policy "$project_id" \
      --flatten="bindings[].members" \
      --filter="bindings.members:$email AND bindings.role:$role" 2>&1 \
      | grep -q bindings; then
        log "$email is missing the role $role"
  fi
}

missing_roles=$(
    print_role_if_missing "$default_cloudbuild_account" roles/secretmanager.secretAccessor &&
    print_role_if_missing "$default_compute_account" roles/secretmanager.secretAccessor &&
    print_role_if_missing "$default_compute_account" roles/dataflow.worker &&
    print_role_if_missing "$default_compute_account" roles/spanner.databaseUser &&
    print_role_if_missing "$default_compute_account" roles/pubsub.editor &&
    print_role_if_missing "$default_compute_account" roles/datapipelines.serviceAgent
)
if [ "$missing_roles" ]; then
  log "Some roles were missing:"
  log "$missing_roles"
  log "Please rerun this after configuring the roles. Exiting..."
  exit 1
else
  log "All roles are configured properly."
fi

# ========================= Enable spanner for project =========================

spanner_state=$(
    gcloud services list --project "$project_id" \
      --filter="name:spanner.googleapis.com" --format="value(state)"
)

if [ "$spanner_state" = "ENABLED" ]; then
  log "Cloud spanner is already enabled for this project."
else
  log "Cloud spanner is disabled. Enabling it..."
  gcloud services enable spanner.googleapis.com --project "$project_id"

  if [ $? -eq 0 ]; then
    log "Cloud Spanner API has been successfully enabled."
  else
    log "Failed to enable the Spanner API."
  fi
fi

# ===================== Create a new Cloud Spanner instance ====================

# Define the Cloud Spanner instance name
instance_name="datasetcreator"
region="us-west1"

# Check if the Cloud Spanner instance exists
if gcloud spanner instances describe "$instance_name" \
      --project "$project_id" &> /dev/null; then
    log "Cloud Spanner instance $instance_name already exists in $project_id."
else
    # Create the Cloud Spanner instance
    log "Creating Cloud Spanner instance $instance_name in $project_id..."
    gcloud spanner instances create "$instance_name" \
        --config="regional-$region" \
        --nodes=1 \
        --project="$project_id" \
        --description="DatasetCreator"

    # Check if the instance creation was successful
    if [ $? -eq 0 ]; then
        log "Cloud Spanner instance $instance_name created successfully."
    else
        log "Failed to create Spanner instance $instance_name in $project_id."
    fi
fi

# ============================== Create a new DB ===============================

db_name="dataset_creator_db"
ddl_file="$(dirname "$0")/database.ddl"

# Check if the database exists within the instance
if gcloud spanner databases describe "$db_name" \
    --instance="$instance_name" \
    --project="$project_id" &> /dev/null; then
    log "Database $db_name already exists in instance $instance_name."
else
    # Create the database within the instance
    log "Creating database $db_name in instance $instance_name..."
    gcloud spanner databases create "$db_name" \
        --instance="$instance_name" \
        --project="$project_id" \
        --ddl-file="$ddl_file"

    # Check if the database creation was successful
    if [ $? -eq 0 ]; then
        log "Database $db_name created successfully in $instance_name."
    else
        log "Failed to create database $db_name in $instance_name."
    fi
fi

# ============================ Create a new bucket =============================

bucket_name="$project_id-dataset-creator"

if gsutil ls -p "$project_id" | grep -q "gs://$bucket_name/" &> /dev/null; then
    log "Bucket $bucket_name already exists in $project_id."
else
    # Create the bucket
    log "Creating bucket $bucket_name in $project_id..."
    gsutil mb -p "$project_id" "gs://$bucket_name"

    # Check if the bucket creation was successful
    if [ $? -eq 0 ]; then
        log "Bucket $bucket_name created successfully in $project_id."
    else
        log "Failed to create bucket $bucket_name in $project_id."
    fi
fi

# ================ Upload GitHub access token to SecretManager =================

secret_name="github_repo_access_token"
gcloud secrets describe "$secret_name" --project="$project_id" &>/dev/null
secret_description_exit_code=$?

# Check if the secret exists
if [ "$do_upgrade" = "false" ] && [ $secret_description_exit_code -eq 0 ]; then
    log "The secret '$secret_name' already exists."
else
    log "Setting secret '$secret_name'..."

    if [ $secret_description_exit_code -ne 0 ]; then
      # Create the secret
      gcloud secrets create "$secret_name" --project="$project_id"
    fi
    # Set the secret's value (can be modified as needed)
    echo "$repo_access_token" | gcloud secrets versions add \
      "$secret_name" --project="$project_id" --data-file=-

    log "The secret '$secret_name' has been set."
fi

# ================== Upload startup script and riegeli patches =================

startup_script_file="$(dirname "$0")/vm_startup_script.sh"

# Check if the file exists locally
if [ ! -f "$startup_script_file" ]; then
    log "Local file does not exist: $startup_script_file"
fi

# Use gcloud to copy the file to the specified bucket
gcloud storage cp "$startup_script_file" "gs://$bucket_name/" \
  --project="$project_id"

# Check if the upload was successful
if [ $? -eq 0 ]; then
    log "Startup script uploaded successfully to gs://$bucket_name/"
else
    log "Failed to upload file to gs://$bucket_name/"
fi

riegeli_patches_dir="$(dirname "$0")/../riegeli_patches"

# Check if the directory exists locally
if [ ! -d "$riegeli_patches_dir" ]; then
    log "Local directory does not exist: $riegeli_patches_dir"
fi

gcloud storage cp -r "$riegeli_patches_dir" "gs://$bucket_name/" \
  --project="$project_id"

if [ $? -eq 0 ]; then
    log "Riegeli patches uploaded successfully to gs://$bucket_name/"
else
    log "Failed to upload patches to gs://$bucket_name/"
fi

# ================ Configure startup script to be the default ==================

gcloud compute project-info add-metadata \
    --metadata "startup-script-url=gs://$bucket_name/vm_startup_script.sh" \
    --project "$project_id"
log "Set the startup script as default for all GCE VMs."

# =============== Add an artifact repository for docker images =================

repository_name="datasetcreator-docker"

if ! gcloud artifacts repositories describe "$repository_name" \
       --project="$project_id" --location="$region" &> /dev/null; then
  log "Creating an artifacts repository named $repository_name..."
  gcloud artifacts repositories create "$repository_name" \
      --repository-format=docker \
      --location="$region" \
      --project="$project_id"

  if [ $? -eq 0 ]; then
    log "Docker repository '$repository_name' added to project '$project_id'."
  else
    log "Could not create docker repository."
  fi
else
  log "The docker artifacts repository already exists."
fi

# =================== Build the beam container for workers =====================

image=$(
  gcloud artifacts docker images list \
    "$region-docker.pkg.dev/$project_id/$repository_name/datasetcreator-image" \
    --format="value(createTime)" --project="$project_id" 2>/dev/null
)

if [ "$do_upgrade" = "false" ] && [ -n "$image" ]; then
  log "Docker image already exists in the docker repository."
else
  log "Building beam workers image..."
  gcloud builds submit --config=setup/gcp/cloudbuild.yaml \
      --project="$project_id"
fi

# ============================== Enable Pub/Sub ================================

# Check if Pub/Sub is enabled in the project
pubsub_enabled=$(
  gcloud services list --project="$project_id" \
    --filter="name:pubsub.googleapis.com" --format="value(state)"
)

if [ "$pubsub_enabled" == "ENABLED" ]; then
  log "Pub/Sub is already enabled in the project $project_id."
else
  # Enable Pub/Sub in the project
  gcloud services enable pubsub.googleapis.com --project="$project_id"

  # Check the result
  if [ $? -eq 0 ]; then
    log "Pub/Sub has been successfully enabled in the project $project_id."
  else
    log "Error: Failed to enable Pub/Sub in the project $project_id."
    log "Please check your permissions and try again."
  fi
fi

# ===================== Create Pub/Sub input/output topics =====================

# Define an array of topic names to check and create
topics=("dataset-creator-unpopulated-queue" "dataset-creator-populated-queue")

for topic_name in "${topics[@]}"; do
  # Check if the Pub/Sub topic exists
  topic_exists=$(
    gcloud pubsub topics describe "projects/$project_id/topics/$topic_name" \
      2>/dev/null
  )

  if [ -z "$topic_exists" ]; then
    # The topic doesn't exist, so create it
    gcloud pubsub topics create "$topic_name" --project="$project_id"

    # Check the result
    if [ $? -eq 0 ]; then
      log "Pub/Sub topic $topic_name has been successfully created."
    else
      log "Error: Failed to create Pub/Sub topic $topic_name."
      log "Please check your permissions and try again."
    fi
  else
    log "Pub/Sub topic $topic_name already exists in the project $project_id."
  fi
done

# ============= Print directions for starting streaming pipelines ==============

streaming_job_name="example-dynamic-population-$project_id"
missing_jobs=()
too_many_features=3

for ((num_features=1; num_features<too_many_features; num_features++)); do
  job_id=$(
    gcloud dataflow jobs list \
      --filter="STATE:Running AND NAME:$streaming_job_name-$num_features" \
      --format="value(JOB_ID)" 2>/dev/null
  )
  if [ -z "$job_id" ]; then
    missing_jobs+=("$num_features")
  fi
done

if [ ${#missing_jobs[@]} -eq 0 ] && [ "$do_upgrade" == "false" ]; then
  log "All streaming jobs are running properly."
elif [ "$do_upgrade" == "true" ]; then
  log "Stopping all running streaming pipelines, since upgrade was requested."
  missing_jobs=()
  for ((num_features=1; num_features<too_many_features; num_features++)); do
    job_id=$(
        gcloud dataflow jobs list \
          --filter="STATE:Running AND NAME:$streaming_job_name-$num_features" \
          --format="value(JOB_ID)" 2>/dev/null
      )
    if [ -n "$job_id" ]; then
      gcloud dataflow jobs cancel "$job_id" --region=us-west1
    fi
    missing_jobs+=("$num_features")
  done
fi

if [ ${#missing_jobs[@]} -ne 0 ]; then
  log "======================================================================="
  log "To complete configuration, run the following in the Verily-AI notebook:"
  log ""
  log "from dataset_creator.dataset_creator.pipeline import dataflow_utils"
  for num_features in "${missing_jobs[@]}"; do
    log "dataflow_utils.run_streaming($num_features)"
  done
  log "======================================================================="
fi
