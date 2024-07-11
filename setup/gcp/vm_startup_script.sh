#!/bin/bash

# ============================ Set script log file =============================

exec >> /var/log/startup.log 2>&1
echo "================ startup at $(date) ================"

log () {
  echo "[**] $(date +%H:%M:%S.%N): $1"
}

# ======================== Setup basic aliases for COS =========================

if grep -iq ID=COS < /etc/os-release; then
  is_cos=true
  log "Running from within a Container-Optimized OS. Setting aliases..."
  shopt -s expand_aliases
  alias apt-get="toolbox apt-get"
  alias dpkg="toolbox dpkg"
  alias gcloud="toolbox gcloud"
  alias gsutil="toolbox gsutil"
else
  is_cos=false
  log "Running from a normal OS, skipping aliases definition."
fi

# ===================== Install required system packages =======================

apt-get update
apt-get install fuse3 wget -y
if [ "$is_cos" == "false" ]; then
  apt-get install libgl1 libglib2.0 python3-pip ca-certificates -y
fi
apt-get clean

if [ "$is_cos" == "true" ]; then
    alias wget="toolbox wget"
fi

# ============================== Install gcsfuse ===============================

pkg_name=gcsfuse_1.0.0_amd64.deb

# Check if gcloud is installed
if ! command -v gcsfuse &> /dev/null; then
    log "gcsfuse is not installed. Installing..."

    # Download the .deb file from the specified URL
    wget "https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v1.0.0/$pkg_name"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        log "Downloaded $pkg_name successfully."

        # Install the .deb file using dpkg
        dpkg -i "$pkg_name"

        # Check if the installation was successful
        if [ $? -eq 0 ]; then
            log "Installed $pkg_name successfully."
            rm -f "$pkg_name"
        else
            log "Failed to install $pkg_name."
        fi
    else
        log "Failed to download $pkg_name."
    fi
else
    log "gcsfuse is already installed."
fi

if [ "$is_cos" == "true" ]; then
    toolbox_root="/var/lib/toolbox/$(ls /var/lib/toolbox)"
    alias gcsfuse="$toolbox_root/usr/bin/gcsfuse"
fi

# ============================= Install gCloud CLI =============================

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    log "Google Cloud SDK is not installed. Installing..."

    # Download and install the Google Cloud SDK
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-444.0.0-linux-x86_64.tar.gz
    tar -xvf google-cloud-cli-444.0.0-linux-x86_64.tar.gz
    chmod +x ./google-cloud-cli/install.sh
    ./google-cloud-cli/install.sh

    # Check installation status
    if [ $? -eq 0 ]; then
        log "Google Cloud SDK has been successfully installed."
    else
        log "Failed to install Google Cloud SDK."
    fi
else
    log "Google Cloud SDK is already installed."
fi

# ============================= Connect to buckets =============================

default_directory="/gcs"
if [ "$is_cos" == "true" ]; then
  # Place all mounts under this specific directory, which is mapped into
  # dataflow worker VMs, in /var/opt/google/gcs.
  target_directory="/var/opt/google/dataflow/gcs"
else
  target_directory="$default_directory"
fi

if [ -z "$project_id" ]; then
  log "project_id not provided from outer context, trying to retrieve it..."
  project_id=$(gcloud config get-value project | tr -d '\n\r\t ')
else
  log "project_id provided from outer context."
fi

# List all buckets in the specified project
buckets=$(gsutil ls -p "$project_id")

# Check if any buckets were found
if [ -z "$buckets" ]; then
  log "No buckets found in '$project_id'."
fi

# Loop through each bucket and attempt to mount it
for bucket in $buckets; do
    # Extract the bucket name from the URL
    bucket=$(echo -n "$bucket" | tr -d '\n\r\t ')
    bucket_name=$(basename "$bucket")

    # Define the mount point directory (change as needed)
    mount_point="$target_directory/$bucket_name"

    # Create the mount point directory if it doesn't exist
    mkdir -p "$mount_point"

    # Attempt to mount the bucket using gcsfuse
    gcsfuse --implicit-dirs -o allow_other --file-mode=777 \
      --dir-mode=777 "$bucket_name" "$mount_point"

    # Check if the mount was successful
    if [ $? -eq 0 ]; then
        log "Mounted bucket $bucket_name to $mount_point."
    else
        log "Failed to mount bucket $bucket_name."
    fi
done

# ================================ Exit if COS =================================

if [ "$is_cos" == "true" ]; then
  # Since we won't be running python directly on the COS itself, time to exit.
  exit
fi

# ========================= Install pyenv system-wide ==========================

python_version=3.9.7

# Check if pyenv is already installed system-wide
if ! command -v /usr/local/pyenv/bin/pyenv &> /dev/null; then

  # Install required dependencies
  apt-get install -y git curl build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev \
        xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

  # Clone the pyenv repository from GitHub to a system-wide location
  git clone https://github.com/pyenv/pyenv.git /usr/local/pyenv

  # Add pyenv to the system-wide environment
  echo 'export PYENV_ROOT="/usr/local/pyenv"' | tee -a /etc/profile.d/pyenv.sh
  echo 'export PATH="$PYENV_ROOT/bin:$PATH"' | tee -a /etc/profile.d/pyenv.sh
  echo 'eval "$(pyenv init - --no-rehash)"' | tee -a /etc/profile.d/pyenv.sh

  # Reload the system-wide environment
  source /etc/profile.d/pyenv.sh

  # Install pyenv-virtualenv (Pyenv plugin for virtual environments)
  git clone https://github.com/pyenv/pyenv-virtualenv.git "$(pyenv root)/plugins/pyenv-virtualenv"

  # Verify the installation
  if command -v pyenv virtualenvs > /dev/null 2>&1; then
      log "Pyenv has been successfully installed system-wide."
  else
      log "Failed to install Pyenv."
  fi

  # Install recommended Python build dependencies
  pyenv install "$python_version"
  pyenv global "$python_version"

  log "Upgrading pip..."
  python -m pip install --upgrade pip
else
  log "pyenv is already installed."
  # Reload the system-wide environment
  source /etc/profile.d/pyenv.sh
fi

# ============== Setup a new jupyter kernel on a new environment ===============

if command -v /opt/conda/bin/jupyter &> /dev/null; then
  kernel_name=dataset-creator
  existing_kernels=$(jupyter kernelspec list)
  dataset_creator_kernel_name="DatasetCreator ($python_version)"

  if echo "$existing_kernels" | grep -q "$kernel_name"; then
    log "The Jupyter kernel $kernel_name exists."
  else
    pip install ipykernel
    log "Setting up a new jupyter kernel named $dataset_creator_kernel_name."
    python -m ipykernel install --name "$kernel_name" --display-name \
        "$dataset_creator_kernel_name" --env PYENV_ROOT /usr/local/pyenv \
        --env PATH "$PATH" --env PYENV_VERSION "$python_version" \
        --env TF_CPP_MIN_LOG_LEVEL "3"
    log "'$dataset_creator_kernel_name' has been added."
  fi
fi

# ============================= Set GitHub access ==============================

if [ -z "$repo_access_token" ]; then
  log "Repo access token not provided from an outer context. Pulling it..."
  repo_access_token=$(
      gcloud secrets versions access latest \
          --secret="github_repo_access_token" --project="$project_id"
  )
else
  log "Repo access token provided from an outer context."
fi

# Trim any whitespace in the access token, in case it exists.
repo_access_token=$(echo -e "$repo_access_token" | tr -d '\n\r\t ')

if [ -n "$repo_access_token" ]; then
  log "Configuring git to use access token instead of SSH keys."
  git config --system \
      url."https://$repo_access_token@github.com/".insteadOf \
      ssh://git@github.com/
fi

# ========================== Install DatasetCreator ============================

# Define the GitHub repository and package name
github_repository="verily-src/dataset-creator"
package_name="dataset-creator"

# Check if the package is installed
if pip show "$package_name" &> /dev/null; then
  log "$package_name is already installed."
else
  log "$package_name is not installed. Installing from GitHub..."

  # Install the package from the GitHub repository using pip.
  export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
  pip install "git+ssh://git@github.com/$github_repository.git"

  # Check if the installation was successful
  if [ $? -eq 0 ]; then
      log "$package_name has been successfully installed from GitHub."
      dataset_creator_file=$(
          python -c "import dataset_creator; print(dataset_creator.__file__)"
      )
      find "$(dirname "$dataset_creator_file")" \
        -type d -name "testdata" -prune -exec rm -rf {} \;
  else
      log "Failed to install $package_name from GitHub."
  fi
fi

# Make the python directories writable so users can pip install into them
chmod -R a+w "$(python -c "import sys; print(sys.prefix)")"

# ============================= Shrink for Docker ==============================

if [ -n "$shrink_pipeline_image" ]; then
  # Delete high-sized directories that are not necessary to run beam in our env.
  # /usr/local/lib/python3.9 is the native python, we install our own pyenv.
  log "Removing unnecessary builtin python."
  rm -rf /usr/local/lib/python3.9
  ln -s /usr/local/pyenv/versions/3.9.7/lib/python3.9 /usr/local/lib/python3.9

  log "Removing unnecessary gcloud installation."
  rm -rf /usr/local/gcloud
fi

# ============================== Install Riegeli ===============================

# Start by updating bazel, as the builtin bazel version in GCP is VERY old

bazel_version=$(
    bazel version | head -n 1 | sed 's/Build label: \([0-9]*\).*/\1/'
)

if [ -z "$bazel_version" ] || (( $bazel_version < 7 )); then
  log "Bazel needs to be upgraded. Doing it now.."
  # Remove the GCP version of bazel
  unlink /usr/local/bin/bazel
  PATH="$PATH"

  apt install apt-transport-https curl gnupg -y
  curl -fsSL https://bazel.build/bazel-release.pub.gpg \
    | gpg --dearmor > /usr/share/keyrings/bazel-archive-keyring.gpg
  echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" \
    | tee /etc/apt/sources.list.d/bazel.list
  apt update && apt install bazel
fi

riegeli_dir=$(python -c "import riegeli; print(riegeli.__file__)")
if [ -z "$patches_dir" ]; then
  patches_dir="$default_directory/$project_id-dataset-creator/riegeli_patches"
else
  log "patches_dir provided from outer context."
fi

if [ -z "$riegeli_dir" ]; then
  log "Riegeli is not installed. Cloning it..."
  git clone https://github.com/google/riegeli.git

  cd riegeli || exit
  git checkout b92772a938146c58b39ae8bda3ca2f978ab01631
  log "Configuring bazel for riegeli installation..."
  ./configure 2>/dev/null

  log "Patching riegeli files..."
  if ! patch WORKSPACE "$patches_dir/WORKSPACE.patch"; then
    log "Could not patch WORKSPACE file."
  fi
  if ! patch python/riegeli/records/BUILD "$patches_dir/BUILD.patch"; then
    log "Could not patch python/riegeli/records/BUILD file."
  fi
  cp -f "$patches_dir/protobuf.patch" third_party/protobuf.patch

  log "Building wheel for installation."
  bazel clean --expunge
  bazel build -c opt //python:build_pip_package --enable_bzlmod=false
  if [ $? -ne 0 ]; then
    log "Could not build build_pip_package.sh.."
  fi
  bazel-bin/python/build_pip_package --dest . --sdist --bdist

  log "Installing riegeli from wheel.."
  pip install riegeli*.whl
  cd ..
  rm -rf riegeli
else
  log "Riegeli is already installed! Skipping installation."
fi

# ============================== Clean pip cache ===============================

pip cache purge

# ============================ Unset GitHub access =============================

log "Returning to SSH keys instead of secret access token."
git config --system \
    --unset url."https://$repo_access_token@github.com/".insteadOf
