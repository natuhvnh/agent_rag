"""
Build and deploy the RAG agent to Azure Container Apps.
Architecture:

  .env
    |
    v
  deployment.py
    |
    +-- Azure Resource Group (pre-existing, verified not created)
    +-- Azure Container Registry (admin user enabled -- authenticates the image pull)
    +-- Container Apps Environment
    +-- Container App (pulls its image using ACR admin credentials)
            |
            +-- Secrets --> Container

The resource group (RG) is pre-existing shared infrastructure -- this script verifies it
exists but does not create it. Its location is inherited for the resources this script
does create, unless LOCATION is set explicitly.

Usage:
    python3 deployment.py

Override configuration via environment variables, e.g.:
    RG=my-existing-rg python3 deployment.py
    APP_NAME=rag-agent-dev python3 deployment.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

SCRIPT_DIR = Path(__file__).resolve().parent

RG = os.environ.get("RG", "3t-agent")
LOCATION = os.environ.get("LOCATION")  # None => inherit from the existing resource group
ACR = os.environ.get("ACR", "ragagentacr")
ENV_NAME = os.environ.get("ENV_NAME", "rag-agent-env")
APP_NAME = os.environ.get("APP_NAME", "rag-agent")
TARGET_PORT = os.environ.get("TARGET_PORT", "8000")
CPU = os.environ.get("CPU", "1.0")
MEMORY = os.environ.get("MEMORY", "2.0Gi")
MIN_REPLICAS = os.environ.get("MIN_REPLICAS", "0")
MAX_REPLICAS = os.environ.get("MAX_REPLICAS", "1")
HTTP_CONCURRENCY = os.environ.get("HTTP_CONCURRENCY", "2")

ENV_FILE = SCRIPT_DIR / ".env"
REQUIRED_SECRETS = [
    "azure_llm_key",
    "embedding_base_url",
    "embedding_key",
    "embedding_deployment",
    "cosmos_url",
    "cosmos_key",
    "tavily_key",
    "rag_api_key",
]


class DeploymentError(Exception):
    """Raised for any unrecoverable deployment failure; caught once in main()."""


def log(message):
    print()
    print(f"==> {message}")


def run(cmd, check=True, capture=True):
    """
    Run a CLI command (az/git/...). Returns stripped stdout when capture=True.

    capture=False streams the child's stdout/stderr straight through -- used for
    `az acr build`, whose live build log is the point of watching it run.
    """
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if check and result.returncode != 0:
            raise DeploymentError(f"{' '.join(cmd)}\n{result.stderr.strip()}")
        return result.stdout.strip()
    else:
        result = subprocess.run(cmd)
        if check and result.returncode != 0:
            raise DeploymentError(f"{' '.join(cmd)} (exit {result.returncode})")
        return ""


def run_ok(cmd):
    """Return True if the command exits 0, without raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_json(cmd):
    return json.loads(run(cmd + ["-o", "json"]))


def image_tag():
    """
    Tag reflects actual content, not just the last commit: `git rev-parse` ignores
    uncommitted changes, which would silently redeploy stale code under a tag ACA
    already has cached (no new revision gets created for an unchanged image string).
    """
    if env_override := os.environ.get("IMAGE_TAG"):
        return env_override

    git_sha = run(["git", "-C", str(SCRIPT_DIR), "rev-parse", "--short", "HEAD"], check=False) or "nogit"
    clean = run_ok(["git", "-C", str(SCRIPT_DIR), "diff", "--quiet"]) and run_ok(
        ["git", "-C", str(SCRIPT_DIR), "diff", "--cached", "--quiet"]
    )
    dirty_suffix = "" if clean else "-dirty"
    timestamp = time.strftime("%Y%m%d%H%M%S")
    return f"{git_sha}{dirty_suffix}-{timestamp}"


def check_prerequisites():
    log("Checking prerequisites")

    if shutil.which("az") is None:
        raise DeploymentError("Azure CLI (az) is not installed.")

    if shutil.which("git") is None:
        raise DeploymentError("git is not installed.")

    run(["az", "extension", "add", "--name", "containerapp", "--upgrade", "--only-show-errors"])

    if not ENV_FILE.is_file():
        raise DeploymentError(f"{ENV_FILE} not found.")


def check_azure_login():
    log("Checking Azure authentication")

    if not run_ok(["az", "account", "show"]):
        raise DeploymentError("Not logged into Azure. Run: az login")

    account = run_json(["az", "account", "show"])
    print(f"Subscription   : {account['name']}")
    print(f"Subscription ID: {account['id']}")


def load_secrets():
    log("Loading environment configuration")

    secrets = dotenv_values(ENV_FILE)

    missing = [key for key in REQUIRED_SECRETS if not secrets.get(key)]
    if missing:
        raise DeploymentError(f"Missing/empty in {ENV_FILE}: {', '.join(missing)}")

    return secrets


def verify_resource_group():
    """
    The resource group is pre-existing shared infrastructure -- this script does not
    create it. Failing loudly here beats silently creating a second group from a typo
    in RG and then filling it with an ACR, environment, identity, and app.

    Returns the group's location, used as the default LOCATION for everything this
    script does create, unless the caller set LOCATION explicitly.
    """
    log("Checking Resource Group")

    if not run_ok(["az", "group", "show", "--name", RG]):
        raise DeploymentError(
            f"Resource group '{RG}' not found in the current subscription.\n"
            f"This script does not create it. Check `az account show`, or set RG=<name>."
        )

    group = run_json(["az", "group", "show", "--name", RG, "--query", "{location:location}"])
    print(f"Resource group : {RG} ({group['location']})")
    return group["location"]


def ensure_acr():
    """
    Admin user enabled -- the Container App authenticates its image pull with ACR admin
    credentials (see get_acr_credentials), not a managed identity. Granting AcrPull via a
    role assignment needs Owner/User Access Administrator; enabling/reading the admin user
    needs only Contributor.
    """
    log("Creating/checking Azure Container Registry")

    if not run_ok(["az", "acr", "show", "--name", ACR, "--resource-group", RG]):
        run(
            [
                "az", "acr", "create",
                "--name", ACR,
                "--resource-group", RG,
                "--location", LOCATION,
                "--sku", "Basic",
                "--admin-enabled", "true",
                "--output", "none",
            ]
        )

    acr_info = run_json(
        ["az", "acr", "show", "--name", ACR, "--resource-group", RG, "--query", "{loginServer:loginServer, adminUserEnabled:adminUserEnabled}"]
    )
    print(f"ACR server: {acr_info['loginServer']}")

    if not acr_info["adminUserEnabled"]:
        log("Enabling ACR admin user")
        run(
            ["az", "acr", "update", "--name", ACR, "--resource-group", RG, "--admin-enabled", "true", "--output", "none"]
        )

    return acr_info["loginServer"]


def get_acr_credentials():
    """
    Admin credentials rather than a managed identity + AcrPull: granting a role
    assignment needs Owner/User Access Administrator, while enabling and reading the
    admin user needs only Contributor -- which is what the deploying account has.
    """
    log("Reading ACR admin credentials")
    creds = run_json(
        ["az", "acr", "credential", "show", "--name", ACR, "--resource-group", RG, "--query", "{username:username, password:passwords[0].value}"]
    )
    print("="*50)
    print(creds["username"], creds["password"])
    return creds["username"], creds["password"]


def build_image(tag):
    """Building in ACR avoids local Apple Silicon ARM64 -> Azure AMD64 problems."""
    log("Building Docker image in Azure Container Registry")
    run(
        ["az", "acr", "build", "--registry", ACR, "--image", tag, "--platform", "linux/amd64", str(SCRIPT_DIR)],
        capture=False,
    )


def ensure_containerapp_env():
    log("Creating/checking Container Apps Environment")
    if not run_ok(["az", "containerapp", "env", "show", "--name", ENV_NAME, "--resource-group", RG]):
        run(
            [
                "az", "containerapp", "env", "create",
                "--name", ENV_NAME,
                "--resource-group", RG,
                "--location", LOCATION,
                "--output", "none",
            ]
        )


def create_or_update_containerapp(acr_server, image, acr_username, acr_password, secrets):
    secret_args = []
    env_var_args = []
    secret_names = {
        "azure_llm_key": "azure-llm-key",
        "embedding_base_url": "embedding-base-url",
        "embedding_key": "embedding-key",
        "embedding_deployment": "embedding-deployment",
        "cosmos_url": "cosmos-url",
        "cosmos_key": "cosmos-key",
        "tavily_key": "tavily-key",
        "rag_api_key": "rag-api-key",
    }
    for env_key, secret_key in secret_names.items():
        secret_args.append(f"{secret_key}={secrets[env_key]}")
        env_var_args.append(f"{env_key}=secretref:{secret_key}")
    env_var_args.append("PYTHONUNBUFFERED=1")

    full_image = f"{acr_server}/{image}"

    if not run_ok(["az", "containerapp", "show", "--name", APP_NAME, "--resource-group", RG]):
        log("Creating Container App")
        run(
            [
                "az", "containerapp", "create",
                "--name", APP_NAME,
                "--resource-group", RG,
                "--environment", ENV_NAME,
                "--image", full_image,
                "--registry-server", acr_server,
                "--registry-username", acr_username,
                "--registry-password", acr_password,
                "--ingress", "external",
                "--target-port", TARGET_PORT,
                "--transport", "auto",
                "--cpu", CPU,
                "--memory", MEMORY,
                "--min-replicas", MIN_REPLICAS,
                "--max-replicas", MAX_REPLICAS,
                "--scale-rule-name", "http-concurrency",
                "--scale-rule-type", "http",
                "--scale-rule-http-concurrency", HTTP_CONCURRENCY,
                "--secrets", *secret_args,
                "--env-vars", *env_var_args,
                "--output", "none",
            ]
        )
    else:
        log("Container App already exists")

        log("Updating registry credentials")
        run(
            ["az", "containerapp", "registry", "set", "--name", APP_NAME, "--resource-group", RG,
             "--server", acr_server, "--username", acr_username, "--password", acr_password, "--output", "none"]
        )

        log("Updating container image")
        run(
            [
                "az", "containerapp", "update",
                "--name", APP_NAME,
                "--resource-group", RG,
                "--image", full_image,
                "--cpu", CPU,
                "--memory", MEMORY,
                "--min-replicas", MIN_REPLICAS,
                "--max-replicas", MAX_REPLICAS,
                "--scale-rule-name", "http-concurrency",
                "--scale-rule-type", "http",
                "--scale-rule-http-concurrency", HTTP_CONCURRENCY,
                "--output", "none",
            ]
        )

        log("Updating Container Apps secrets")
        run(
            ["az", "containerapp", "secret", "set", "--name", APP_NAME, "--resource-group", RG, "--secrets", *secret_args, "--output", "none"]
        )

        log("Updating environment variables")
        run(
            ["az", "containerapp", "update", "--name", APP_NAME, "--resource-group", RG, "--set-env-vars", *env_var_args, "--output", "none"]
        )

    # Health probes: `az containerapp create`/`update` have no
    # --startup-probe/--readiness-probe/--liveness-probe flags (probes are only
    # configurable via ARM template or `--yaml`, and `--yaml` replaces the whole
    # object -- see https://learn.microsoft.com/en-us/azure/container-apps/health-probes
    # and https://github.com/microsoft/azure-container-apps/issues/516). This relies
    # on the ACA default probes instead: with ingress enabled, the default startup
    # probe is TCP on the target port with a ~240s failure budget (240 attempts x 1s),
    # comfortably longer than this app's ~10-20s cold start. Add a proper HTTP probe
    # via Bicep/YAML if you need /healthz itself checked rather than just the TCP port.


def get_fqdn():
    log("Getting application URL")
    fqdn = run(
        ["az", "containerapp", "show", "--name", APP_NAME, "--resource-group", RG, "--query", "properties.configuration.ingress.fqdn", "-o", "tsv"]
    )
    if not fqdn:
        raise DeploymentError("Unable to retrieve Container App FQDN.")
    return f"https://{fqdn}"


def wait_for_health(url, max_attempts=30, sleep_seconds=10):
    log("Waiting for application health")

    for attempt in range(1, max_attempts + 1):
        print(f"Health check {attempt}/{max_attempts}...")
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=10) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(sleep_seconds)

    return False


def print_summary(acr_server, image, url, healthy):
    print()
    print("=" * 60)
    print("Deployment completed")
    print("=" * 60)
    print()
    print(f"Resource Group : {RG}")
    print(f"Location       : {LOCATION}")
    print(f"ACR            : {acr_server}")
    print(f"Environment    : {ENV_NAME}")
    print(f"Container App  : {APP_NAME}")
    print(f"Image          : {image}")
    print(f"URL            : {url}")
    print()

    if healthy:
        print("Health         : OK")
        print()
        print("Health endpoint:")
        print(f"  {url}/healthz")
        print()
        print("Test the RAG agent:")
        print()
        print(f"curl -X POST {url}/ask \\")
        print("  -H 'content-type: application/json' \\")
        print("  -H \"x-api-key: $rag_api_key\" \\")
        print("  -d '{\"question\":\"What is Hogwarts?\"}'")
        print()
        print("Note: ACA ingress caps a request at 240s. A cold, multi-step agent run can")
        print("approach that on the blocking call above -- prefer streaming for long questions:")
        print()
        print(f"curl -N -X POST {url}/ask \\")
        print("  -H 'content-type: application/json' \\")
        print("  -H 'accept: text/event-stream' \\")
        print("  -H \"x-api-key: $rag_api_key\" \\")
        print("  -d '{\"question\":\"What is Hogwarts?\"}'")
        print()
    else:
        print("Health         : FAILED")
        print()
        print("The container was deployed but /healthz did not return HTTP 200.")
        print()
        print("Recent logs:")
        print()
        run(["az", "containerapp", "logs", "show", "--name", APP_NAME, "--resource-group", RG, "--tail", "100"], check=False, capture=False)


def main():
    global LOCATION

    check_prerequisites()
    check_azure_login()
    secrets = load_secrets()

    tag = image_tag()
    image = f"rag-agent:{tag}"
    print(f"Image         : {image}")

    group_location = verify_resource_group()
    if LOCATION is None:
        LOCATION = group_location

    acr_server = ensure_acr()
    build_image(image)
    ensure_containerapp_env()
    acr_username, acr_password = get_acr_credentials()
    create_or_update_containerapp(acr_server, image, acr_username, acr_password, secrets)

    url = get_fqdn()
    healthy = wait_for_health(url)
    print_summary(acr_server, image, url, healthy)

    if not healthy:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except DeploymentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
