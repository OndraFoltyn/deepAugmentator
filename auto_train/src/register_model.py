import mlflow
from mlflow.tracking import MlflowClient
import argparse

def parse_args():
    """
    Parse command-line arguments for model registration.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--tracking_uri", type=str, default="http://192.168.40.2:5000/", help="MLflow tracking server URI")
    parser.add_argument("--model_uri", type=str, default="runs:/ebbf0e04724347ffb8c2c641281ece54/model", help="URI of the model to register")
    parser.add_argument("--model_name", type=str, default="Testing_model_registration", help="Name of the model to register")
    parser.add_argument("--model_alias", type=str, default="production", help="Alias to assign to the model version")
    parser.add_argument("--model_tag_key", type=str, default="task", help="Tag key to assign to the model version")
    parser.add_argument("--model_tag_value", type=str, default="ner", help="Tag value to assign to the model version")

    return parser.parse_args()


def initialize_mlflow_client(tracking_uri: str) -> MlflowClient:
    """
    Initialize and return an MLflow client with the specified tracking URI.
    """
    mlflow.set_tracking_uri(uri=tracking_uri)
    return MlflowClient()


def model_exists(client: MlflowClient, model_name: str) -> bool:
    """
    Check if a model exists in the MLflow Registry.
    """
    try:
        return len(client.search_model_versions(f"name='{model_name}'")) > 0
    except mlflow.exceptions.MlflowException as e:
        print(f"Error while checking model existence: {e}")
        return False


def register_model(
        args,
        client: MlflowClient
    ):
    """
    Register a model in the MLflow Model Registry.
    """
    registered_model = model_exists(client, args.model_name)
    action = "new version" if registered_model else "new model"
    print(f"\nPreparing to register {action} '{args.model_name}' from URI: {args.model_uri}")

    try:
        model_version = mlflow.register_model(model_uri=args.model_uri, name=args.model_name)
    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow exception occurred during registration: {e}")
        return
    
    print(f"\nApplying alias '{args.model_alias}' and tag '{args.model_tag_key}={args.model_tag_value}'...")

    try:
        client.set_registered_model_alias(
            name=args.model_name,
            version=model_version.version,
            alias=args.model_alias,
        )

        client.set_model_version_tag(
            name=args.model_name,
            version=model_version.version,
            key=args.model_tag_key,
            value=args.model_tag_value,
        )

    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow exception occurred while setting '{args.model_alias}' alias: {e}")
    

def print_model_versions(args, client: MlflowClient):
    """Print all versions of a registered model."""
    try:
        versions = client.search_model_versions(f"name='{args.model_name}'")
    except mlflow.exceptions.MlflowException as e:
        print(f"Failed to fetch model versions: {e}")
        return

    print(f"\nModel Versions for '{args.model_name}':")
    for mv in versions:
        print(
            f"Version: {mv.version}, Stage: {mv.current_stage}, "
            f"Alias: {mv.aliases}, Tags: {mv.tags}, Source: {mv.source}"
        )


# Example usage
if __name__ == "__main__":
    args = parse_args()

    client = initialize_mlflow_client(tracking_uri=args.tracking_uri)

    register_model(args, client)
    print_model_versions(args, client)