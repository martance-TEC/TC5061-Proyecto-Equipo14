"""La copia operativa conserva un alias explícito y no promueve otro modelo."""
from unittest.mock import Mock

import pytest

from mibici_equipo14.operations import MODEL_NAME, define_operating_version


def test_real_version_gets_local_champion_once():
    client = Mock()
    client.get_registered_model.return_value.aliases = {}
    define_operating_version(client, '1', {'synthetic_ci': False})
    client.set_registered_model_alias.assert_called_once_with(MODEL_NAME, 'champion', '1')
    client.get_registered_model.return_value.aliases = {'champion': '1'}
    client.reset_mock()
    define_operating_version(client, '1', {'synthetic_ci': False})
    client.set_registered_model_alias.assert_not_called()


def test_other_champion_requires_review_without_changes():
    client = Mock()
    client.get_registered_model.return_value.aliases = {'champion': '2'}
    with pytest.raises(ValueError, match='otra versión'):
        define_operating_version(client, '1', {'synthetic_ci': False})
    client.set_registered_model_alias.assert_not_called()
    client.set_model_version_tag.assert_not_called()


def test_synthetic_ci_model_is_not_declared_champion():
    client = Mock()
    define_operating_version(client, '1', {'synthetic_ci': True})
    assert client.mock_calls == []
