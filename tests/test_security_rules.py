"""La revisión no muestra credenciales y respeta los archivos excluidos."""
import pytest

from mibici_equipo14.quality_security import credential_findings, forbidden_path


@pytest.mark.parametrize('path', ['.env', '.env.local', 'data/raw/trips.csv',
                                 'mlartifacts/model/MLmodel', 'models/model.pkl', 'mlflow.db'])
def test_sensitive_files_must_not_be_tracked(path):
    assert forbidden_path(path)


@pytest.mark.parametrize('path', ['.env.example', 'data/raw/.gitkeep',
                                 'models/.gitkeep', 'references/data_manifest.json'])
def test_public_configuration_and_manifests_are_allowed(path):
    assert not forbidden_path(path)


def test_token_detection_returns_only_the_rule():
    token = 'ghp_' + 'a' * 36
    assert credential_findings(token) == ['github_token']
    assert token not in str(credential_findings(token))


def test_environment_variable_reference_is_not_a_credential():
    assert credential_findings('password = os.environ["DATABASE_PASSWORD"]') == []
