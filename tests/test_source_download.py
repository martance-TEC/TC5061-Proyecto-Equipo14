"""La ingesta solo descarga fuentes oficiales y no conserva archivos incompletos."""
import pytest

from mibici_equipo14 import dataset


@pytest.mark.parametrize('url', [
    'file:///etc/passwd', 'http://mibici.net/trips.csv',
    'https://example.org/trips.csv', 'https://mibici.net.example.org/trips.csv',
    'https://user:secret@mibici.net/trips.csv', 'https://mibici.net:8443/trips.csv',
])
def test_untrusted_source_never_reaches_the_network(tmp_path, monkeypatch, url):
    def unexpected_request(*args, **kwargs):
        pytest.fail('Una fuente rechazada no debe abrir una conexión')
    monkeypatch.setattr(dataset.requests, 'get', unexpected_request)
    with pytest.raises(ValueError, match='HTTPS'):
        dataset.download_source(url, tmp_path / 'trips.csv')
    assert not (tmp_path / 'trips.csv').exists()


@pytest.mark.parametrize('status', [200, 302])
def test_download_accepts_only_a_complete_response(tmp_path, monkeypatch, status):
    class Response:
        status_code = status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'trip_id\n1\n'

    def request(url, **options):
        assert options['allow_redirects'] is False
        assert options['timeout'] == (10, 120)
        return Response()

    monkeypatch.setattr(dataset.requests, 'get', request)
    path = tmp_path / 'trips.csv'
    if status == 200:
        dataset.download_source('https://mibici.net/trips.csv', path)
        assert path.read_bytes() == b'trip_id\n1\n'
    else:
        with pytest.raises(ValueError, match='redirecciones'):
            dataset.download_source('https://mibici.net/trips.csv', path)
        assert not path.exists()
    assert not path.with_suffix('.csv.part').exists()


def test_interrupted_download_keeps_the_previous_file(tmp_path, monkeypatch):
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b'incomplete'
            raise OSError('Conexión interrumpida')

    monkeypatch.setattr(dataset.requests, 'get', lambda *a, **k: Response())
    path = tmp_path / 'trips.csv'
    path.write_bytes(b'previous')
    with pytest.raises(OSError, match='interrumpida'):
        dataset.download_source('https://mibici.net/trips.csv', path)
    assert path.read_bytes() == b'previous'
    assert not path.with_suffix('.csv.part').exists()
