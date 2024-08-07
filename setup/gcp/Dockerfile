# CMD to build and deploy the docker to GCP Artifact Registry:
#   gcloud builds submit --config=setup/gcp/cloudbuild.yaml

# Guide used to setup docker:
#   https://cloud.google.com/build/docs/build-push-docker-image
# Guide used to allow access to pip install from github repo:
#   https://cloud.google.com/build/docs/access-github-from-build

FROM apache/beam_python3.9_sdk:2.56.0

ARG PROJECT_ID
ENV project_id=${PROJECT_ID}
ENV patches_dir=/opt/app/
ENV shrink_pipeline_image=true

COPY gcp/vm_startup_script.sh /opt/app/startup.sh
COPY riegeli_patches/* /opt/app/
RUN chmod +x /opt/app/startup.sh
RUN /bin/bash -c "/opt/app/startup.sh"

# The buckets are mounted to /var/opt/google/dataflow/gcs on the worker VM host.
# This path is mapped to /var/opt/google/gcs on the container, so we symlink
# /gcs to that path, so we always have all buckets under /gcs.
RUN /bin/bash -c "ln -s /var/opt/google/gcs /gcs"

# Define the new entrypoint
RUN echo "#!/bin/bash"                    >> /opt/app/entrypoint.sh && \
    echo 'source /etc/profile.d/pyenv.sh' >> /opt/app/entrypoint.sh && \
    echo '/opt/apache/beam/boot "$@"'     >> /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh
ENTRYPOINT [ "/opt/app/entrypoint.sh" ]
