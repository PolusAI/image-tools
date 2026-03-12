import argparse

from kubernetes import client
from kubernetes import config


def setup_k8s_api():
    """Common actions to setup Kubernetes API access to Argo workflows."""
    config.load_incluster_config()  # Only works inside of JupyterLab Pod

    return client.CustomObjectsApi()


api_instance = setup_k8s_api()

group = "argoproj.io"  # str | The custom resource's group name
version = "v1alpha1"  # str | The custom resource's version
namespace = "default"  # str | The custom resource's namespace
plural = "workflows"  # str | The custom resource's plural name. For TPRs this would be lowercase plural kind.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main", description="Build Docker Container")

    # Add command-line argument for plugin name, docker repo and version
    parser.add_argument(
        "--plugin_name", dest="plugin_name", type=str, help="Plugin Name", required=True,
    )
    parser.add_argument(
        "--docker_hub_repo",
        dest="docker_hub_repo",
        type=str,
        help="Docker Repo Name",
        required=True,
    )
    parser.add_argument(
        "--version",
        dest="version",
        type=str,
        help="Docker Image Version",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()
    plugin_name = args.plugin_name
    docker_hub_repo = args.docker_hub_repo
    docker_version = args.version

    generated_name = f"build-polus-imagej-{plugin_name}"
    print(f"generated_name: {generated_name}")

    subpath = f"temp/plugins/polus-imagej-{plugin_name}"
    print(f"subpath: {subpath}")

    destination = f"polusai/{docker_hub_repo}:{docker_version}"
    print(f"destination: {destination}")

    body = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"generateName": generated_name},
        "spec": {
            "entrypoint": "kaniko",
            "volumes": [
                {
                    "name": "kaniko-secret",
                    "secret": {
                        "secretName": "labshare-docker",
                        "items": [{"key": ".dockerconfigjson", "path": "config.json"}],
                    },
                },
                {
                    "name": "workdir",
                    "persistentVolumeClaim": {"claimName": "wipp-pv-claim"},
                },
            ],
            "templates": [
                {
                    "name": "kaniko",
                    "container": {
                        "image": "gcr.io/kaniko-project/executor:latest",
                        "args": [
                            "--dockerfile=/workspace/Dockerfile",
                            "--context=dir:///workspace",
                            f"--destination={destination}",
                        ],
                        "volumeMounts": [
                            {
                                "name": "kaniko-secret",
                                "mountPath": "/kaniko/.docker",
                            },
                            {
                                "name": "workdir",
                                "mountPath": "/workspace",
                                "subPath": subpath,
                            },
                        ],
                    },
                },
            ],
        },
    }

    api_response = api_instance.create_namespaced_custom_object(
        group, version, namespace, plural, body,
    )
