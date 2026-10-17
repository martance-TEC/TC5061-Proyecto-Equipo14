import pytest


def test_source_hash_changes_when_content_changes(tmp_path):
    from mibici_equipo14.dataset import sha256
    path=tmp_path/'source.csv'
    path.write_text('trip_id\n1\n',encoding='utf-8')
    old=sha256(path)
    path.write_text('trip_id\n2\n',encoding='utf-8')
    assert old != sha256(path)
