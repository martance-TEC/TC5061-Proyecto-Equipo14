"""Chequeos locales de archivos, credenciales y código Python."""
import json
from pathlib import Path
import re
import shutil
import subprocess  # nosec B404: argumentos en listas, ejecutables locales y sin shell.
import sys

from .config import SETTINGS as S

TOKEN_RULES = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{80,255})\b'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'literal_credential': re.compile(
        r'''(?im)^\s*(?:password|passwd|api_key|client_secret|access_token)\s*=\s*["']([^"'\n]{8,})["']'''),
}
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.csv',
                 '.env', '.ipynb', '.ini', '.cfg', '.xml', '.html', '.sh', '.ps1'}


def credential_findings(content):
    """Devuelve reglas, nunca los valores encontrados."""
    return [name for name, pattern in TOKEN_RULES.items() if pattern.search(content)]


def forbidden_path(name):
    path = Path(name)
    if path.name == '.gitkeep' or path.name == '.env.example':
        return False
    return (path.parts[0] in {'data', 'models', 'mlruns', 'mlartifacts'}
            or path.name.startswith('.env')
            or path.suffix in {'.db', '.sqlite', '.sqlite3', '.parquet', '.pkl', '.joblib'})


def git_output(*arguments):
    git = shutil.which('git')
    if not git:
        raise RuntimeError('Git no está disponible')
    return subprocess.check_output(  # nosec B603: Git resuelto, sin shell ni órdenes externas.
        [git, *arguments], cwd=S.root)


def repository_scan():
    """Revisa archivos actuales e historial local completo, sin mostrar secretos."""
    files = git_output('ls-files', '-z').decode().split('\0')
    findings = []
    for name in filter(None, files):
        if forbidden_path(name):
            findings.append({'path': name, 'rule': 'excluded_file_tracked'})
        path = S.root / name
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            for rule in credential_findings(path.read_text(encoding='utf-8', errors='replace')):
                findings.append({'path': name, 'rule': rule})
    entries = git_output('rev-list', '--objects', '--all').decode().splitlines()
    historical = [(line.split(' ', 1)[0], line.split(' ', 1)[1]) for line in entries
                  if ' ' in line]
    git = shutil.which('git')
    types = subprocess.check_output(  # nosec B603: consulta fija de objetos Git locales.
        [git, 'cat-file', '--batch-check=%(objecttype)'], cwd=S.root,
        input=('\n'.join(oid for oid, _ in historical) + '\n').encode()).decode().splitlines()
    scanned = 0
    for (oid, name), object_type in zip(historical, types):
        if object_type != 'blob':
            continue
        if forbidden_path(name):
            findings.append({'path': name, 'object': oid, 'rule': 'excluded_file_in_history'})
        if Path(name).suffix not in TEXT_SUFFIXES:
            continue
        # cat-file por objeto evita interpretar rutas o contenido como una orden.
        content = git_output('cat-file', 'blob', oid).decode('utf-8', errors='replace')
        scanned += 1
        for rule in credential_findings(content):
            findings.append({'path': name, 'object': oid, 'rule': rule})
    return {'passed': not findings, 'tracked_files': len(list(filter(None, files))),
            'historical_text_objects': scanned, 'findings': findings,
            'scope': 'Reglas de credenciales en texto; no analiza contenido comprimido de imágenes, PDF o Word'}


def run(output, audit_dependencies=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scan = repository_scan()
    bandit_path = output / 'bandit.json'
    command = [sys.executable, '-m', 'bandit', '-r', 'mibici_equipo14', '-f', 'json',
               '-o', str(bandit_path)]
    result = subprocess.run(command, cwd=S.root, capture_output=True, text=True)  # nosec B603: orden fija, sin shell.
    if not bandit_path.exists() or result.returncode not in (0, 1):
        raise RuntimeError('Bandit no completó el análisis: ' + result.stderr)
    report = json.loads(bandit_path.read_text(encoding='utf-8'))
    # Las llamadas locales revisadas se justifican en la línea, no con una exclusión global.
    summary = {'repository': scan, 'bandit': {'returncode': result.returncode,
               'findings': len(report['results']), 'errors': report['errors']}}
    summary['passed'] = scan['passed'] and not report['results'] and not report['errors']
    if audit_dependencies:
        audit_path = output / 'pip_audit.json'
        command = [sys.executable, '-m', 'pip_audit', '--local', '--format', 'json',
                   '--output', str(audit_path), '--progress-spinner', 'off']
        result = subprocess.run(command, cwd=S.root, capture_output=True, text=True)  # nosec B603: orden fija, sin shell.
        if not audit_path.exists() or result.returncode not in (0, 1):
            raise RuntimeError('pip-audit no completó el análisis: ' + result.stderr)
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        vulnerabilities = sum(len({v['id'] for v in d.get('vulns', [])})
                              for d in audit['dependencies'])
        summary['pip_audit'] = {'returncode': result.returncode,
                                'vulnerabilities': vulnerabilities,
                                'affected_packages': sum(bool(d.get('vulns')) for d in audit['dependencies'])}
        summary['passed'] = summary['passed'] and result.returncode == 0
    else:
        summary['pip_audit'] = {'status': 'No solicitado; ejecutar security --audit-dependencies'}
    (output / 'security.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Repositorio: {"APROBADO" if scan["passed"] else "CON HALLAZGOS"}')
    print(f'Bandit: {len(report["results"])} hallazgos, {len(report["errors"])} errores')
    if audit_dependencies:
        print(f'pip-audit: {summary["pip_audit"]["vulnerabilities"]} vulnerabilidades conocidas')
    return summary['passed']
