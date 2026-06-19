"""Validated, read-only S3 downloads for API inference."""

import os
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from src.config import load_config


AWS_CONFIG_PATH = Path("configs") / "aws.yaml"


class S3InputError(ValueError):
    """The requested S3 object does not satisfy API input rules."""


class S3AccessDeniedError(Exception):
    """AWS denied access to the requested object."""


class S3ObjectNotFoundError(Exception):
    """The requested S3 object does not exist."""


class AWSConfigurationError(Exception):
    """AWS configuration or credentials are unavailable."""


class S3DownloadError(Exception):
    """An unexpected S3 download failure occurred."""


def _load_environment() -> None:
    """Load local environment values without overriding existing variables."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise AWSConfigurationError(
            "python-dotenv unavailable; install project requirements."
        ) from exc

    load_dotenv()


def _load_s3_config() -> dict:
    """Load S3 settings from YAML and user-specific values from the environment."""

    try:
        _load_environment()
        config = load_config(AWS_CONFIG_PATH)
        s3_config = dict(config["s3"])
        required_keys = {
            "allowed_prefixes",
            "download_dir",
            "max_object_size_mb",
        }
        missing_keys = required_keys.difference(s3_config)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Missing S3 configuration values: {missing}")
        bucket_name = os.getenv("AWS_BUCKET_NAME", "").strip()
        if not bucket_name:
            raise ValueError("AWS_BUCKET_NAME environment variable is required.")
        if not s3_config["allowed_prefixes"]:
            raise ValueError("S3 prefix allowlist must not be empty.")
        if float(s3_config["max_object_size_mb"]) <= 0:
            raise ValueError("S3 maximum object size must be greater than zero.")
        s3_config["region"] = os.getenv("AWS_REGION", "ca-west-1").strip() or "ca-west-1"
        s3_config["allowed_buckets"] = [bucket_name]
        return s3_config
    except AWSConfigurationError:
        raise
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise AWSConfigurationError(f"AWS configuration unavailable: {exc}") from exc


def _validate_location(
    bucket: str,
    key: str,
    s3_config: dict,
    supported_extensions: set[str],
) -> str:
    """Validate an S3 location and return its normalized extension."""

    if bucket not in s3_config["allowed_buckets"]:
        raise S3InputError("S3 bucket is not allowlisted.")
    if not any(key.startswith(prefix) for prefix in s3_config["allowed_prefixes"]):
        raise S3InputError("S3 key does not start with an allowed prefix.")

    filename = Path(key).name
    extension = Path(filename).suffix.lower().lstrip(".")
    if not filename or extension not in supported_extensions:
        allowed = ", ".join(sorted(supported_extensions))
        raise S3InputError(f"Unsupported image type. Allowed extensions: {allowed}.")
    return extension


def _raise_for_client_error(exc: Exception) -> NoReturn:
    """Convert common AWS client errors into API-facing domain errors."""

    response = getattr(exc, "response", {})
    error = response.get("Error", {})
    code = str(error.get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if code in {"AccessDenied", "403"} or status == 403:
        raise S3AccessDeniedError("Access denied for the requested S3 object.") from exc
    if code in {"NoSuchKey", "NotFound", "404"} or status == 404:
        raise S3ObjectNotFoundError("S3 object not found.") from exc
    raise S3DownloadError("S3 download failed.") from exc


def download_s3_image(
    bucket: str,
    key: str,
    supported_extensions: set[str],
) -> Path:
    """Validate and download one S3 image to a unique local path."""

    s3_config = _load_s3_config()
    extension = _validate_location(bucket, key, s3_config, supported_extensions)
    target_path = None
    download_complete = False

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        client = boto3.client("s3", region_name=s3_config["region"])
        metadata = client.head_object(Bucket=bucket, Key=key)
        object_size = int(metadata.get("ContentLength", 0))
        max_size = float(s3_config["max_object_size_mb"]) * 1024 * 1024
        if object_size > max_size:
            raise S3InputError(
                f"S3 object exceeds the {s3_config['max_object_size_mb']} MB limit."
            )

        download_dir = Path(s3_config["download_dir"])
        download_dir.mkdir(parents=True, exist_ok=True)
        target_path = download_dir / f"{uuid4().hex}.{extension}"
        client.download_file(bucket, key, str(target_path))
        download_complete = True
        return target_path
    except ImportError as exc:
        raise AWSConfigurationError("AWS SDK unavailable; install project requirements.") from exc
    except ClientError as exc:
        _raise_for_client_error(exc)
    except NoCredentialsError as exc:
        raise AWSConfigurationError("AWS credentials unavailable.") from exc
    except S3InputError:
        raise
    except (BotoCoreError, OSError, ValueError) as exc:
        raise S3DownloadError("S3 download failed.") from exc
    finally:
        if target_path is not None and not download_complete:
            target_path.unlink(missing_ok=True)
