"""
Deploy the RAG agent to Azure AI Foundry as a hosted agent (Responses protocol).

Uploads main.py + the graph modules + embedding/bm25.pkl as a code zip, has the service
remote-build it against requirements-foundry.txt, waits for the new version to go active,
then routes 100% of the agent's traffic to it and leaves it running. Run with:
    python3 deploy_azure_foundry.py
"""

import os
import tempfile
import time
import zipfile
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
  AgentEndpointConfig,
  CodeConfiguration,
  CodeDependencyResolution,
  FixedRatioVersionSelectionRule,
  HostedAgentDefinition,
  ProtocolConfiguration,
  ProtocolVersionRecord,
  ResponsesProtocolConfiguration,
  VersionSelector,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import dotenv_values, load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"

endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
agent_name = os.environ.get("FOUNDRY_HOSTED_AGENT_NAME", "basic-agent")

RUNTIME_FILES = [
  "main.py",
  "graphs.py",
  "nodes.py",
  "dependencies.py",
  "models.py",
  "prompts.py",
  "helper_functions.py",
  "vecstore.py",
]

RUNTIME_DATA_FILES = [
  "embedding/bm25.pkl",
]

REQUIRED_SECRETS = [
  "azure_llm_key",
  "embedding_base_url",
  "embedding_key",
  "embedding_deployment",
  "cosmos_url",
  "cosmos_key",
  "tavily_key",
]


def create_code_zip() -> Path:
  zip_path = Path(tempfile.gettempdir()) / f"{agent_name}.zip"

  with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
    for name in RUNTIME_FILES + RUNTIME_DATA_FILES:
      zip_file.write(SCRIPT_DIR / name, name)
    zip_file.write(SCRIPT_DIR / "requirements-foundry.txt", "requirements.txt")
  return zip_path


def load_environment_variables() -> dict:
  secrets = dotenv_values(ENV_FILE)
  missing = [key for key in REQUIRED_SECRETS if not secrets.get(key)]
  if missing:
    raise RuntimeError(
      f"Missing/empty required secret(s) in {ENV_FILE}: {', '.join(missing)}"
    )
  return {key: secrets[key] for key in REQUIRED_SECRETS}


def wait_for_active_version(project_client: AIProjectClient, version: str) -> None:
  for attempt in range(60):
    time.sleep(10)
    details = project_client.agents.get_version(
      agent_name=agent_name,
      agent_version=version,
    )
    status = details["status"]
    print(f"Provisioning status: {status} (attempt {attempt + 1}/60)")

    if status == "active":
      return

    if status == "failed":
      # The remote pip build's final error line lands in details["error"]["message"];
      # the full details dict otherwise buries it.
      error = details.get("error") or {}
      raise RuntimeError(
        f"Hosted agent provisioning failed: {error.get('message') or dict(details)}"
      )

  raise RuntimeError("Timed out waiting for the hosted agent version to become active.")


def main():
  environment_variables = load_environment_variables()
  code_zip_path = create_code_zip()

  with (
    code_zip_path.open("rb") as code_stream,
    DefaultAzureCredential() as credential,
    AIProjectClient(endpoint=endpoint, credential=credential, allow_preview=True) as project_client,
  ):
    try:
      created = project_client.agents.create_version_from_code(
        agent_name=agent_name,
        description="RAG plan-and-execute agent, deployed from local Python source.",
        definition=HostedAgentDefinition(
          cpu="1",
          memory="2Gi",
          code_configuration=CodeConfiguration(
            runtime="python_3_13",
            entry_point=["python", "main.py"],
            dependency_resolution=CodeDependencyResolution.REMOTE_BUILD,
          ),
          environment_variables=environment_variables,
          protocol_versions=[
            ProtocolVersionRecord(protocol="responses", version="2.0.0")
          ],
        ),
        code=code_stream,
      )
    except ResourceNotFoundError as exc:
      raise RuntimeError(
        f"Agent '{agent_name}' does not exist yet -- create it first "
        f"(project_client.agents.create_version(...)) before deploying a new version."
      ) from exc

    print(f"Created hosted agent version {created.version}")

    wait_for_active_version(project_client, created.version)

    project_client.agents.update_details(
      agent_name=agent_name,
      agent_endpoint=AgentEndpointConfig(
        version_selector=VersionSelector(
          version_selection_rules=[
            FixedRatioVersionSelectionRule(
              agent_version=created.version,
              traffic_percentage=100,
            ),
          ]
        ),
        protocol_configuration=ProtocolConfiguration(
          responses=ResponsesProtocolConfiguration()
        ),
      ),
    )

    print(f"Agent endpoint configured for version {created.version}")

    with project_client.get_openai_client(agent_name=agent_name) as openai_client:
      # Stream: a plan-and-execute run takes minutes, so print progress rather than
      # appearing to hang for the whole smoke test.
      stream = openai_client.responses.create(
        input="Search the web and give me the answer for question How Event Order API (POS) works ?",
        stream=True,
      )
      output_text = ""
      for event in stream:
        if event.type == "response.output_text.delta":
          print(event.delta, end="", flush=True)
          output_text += event.delta
      print()

    if not output_text:
      raise RuntimeError("Smoke test produced no answer text -- check the session logs.")

    agent = project_client.agents.get(agent_name=agent_name)
    print(f"Deployed and live: version {created.version}")
    print(f"Agent endpoint: {agent.agent_endpoint}")


if __name__ == "__main__":
  main()
