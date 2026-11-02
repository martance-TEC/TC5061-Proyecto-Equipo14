"""Ejecuta controles de calidad y conserva sus resultados locales."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess  # nosec B404: comandos fijos con el intérprete actual, sin shell.
import sys

from .config import SETTINGS as S


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=['tests', 'data', 'security', 'all', 'summary', 'evidence'])
    parser.add_argument('--output', type=Path, default=S.root / 'data/quality')
    parser.add_argument('--audit-dependencies', action='store_true',
                        help='Consulta la base de vulnerabilidades; requiere internet')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output / 'matplotlib'))
    os.environ.setdefault('DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    if args.check == 'summary':
        show_summary(args.output)
        return 0
    if args.check == 'evidence':
        write_evidence(args.output)
        return 0
    if S.root in args.output.parents:
        display = args.output.relative_to(S.root).as_posix()
    else:
        display = args.output.name
    checks = ['tests', 'data', 'security'] if args.check == 'all' else [args.check]
    summary = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    from .quality_security import git_output
    summary['source_commit'] = git_output('rev-parse', 'HEAD').decode().strip()
    summary['working_tree_dirty'] = bool(git_output('status', '--porcelain'))
    for check in checks:
        print(f'\n=== {check.upper()} ===', flush=True)
        try:
            if check == 'tests':
                command = [sys.executable, '-m', 'pytest', '-q', '--tb=short',
                           '--junitxml=' + str(args.output / 'pytest.xml'),
                           '--cov=mibici_equipo14', '--cov-branch',
                           '--cov-report=json:' + str(args.output / 'coverage.json'),
                           '--cov-report=term-missing:skip-covered']
                result = subprocess.run(command, cwd=S.root, capture_output=True, text=True)  # nosec B603: intérprete actual y argumentos fijos.
                (args.output / 'pytest.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
                print(result.stdout, flush=True)
                if result.returncode:
                    print(result.stderr, flush=True)
                passed = result.returncode == 0
            elif check == 'data':
                from .quality_data import run
                passed = run(args.output)
            else:
                from .quality_security import run
                passed = run(args.output, args.audit_dependencies)
            summary['checks'][check] = {'passed': passed}
        except Exception as error:
            summary['checks'][check] = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
            print(f'{check}: {type(error).__name__}: {error}', flush=True)
    summary['passed'] = all(c['passed'] for c in summary['checks'].values())
    (args.output / f'{args.check}_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\nResultado: {"APROBADO" if summary["passed"] else "REQUIERE REVISIÓN"}')
    print(f'Resultados locales: {display}')
    return 0 if summary['passed'] else 1


def show_summary(output):
    """Muestra resultados guardados; no sustituye una ejecución de los controles."""
    from defusedxml import ElementTree

    print('MiBici | Calidad y seguridad')
    path = output / 'all_summary.json'
    if not path.exists():
        raise FileNotFoundError('Falta ejecutar quality all')
    summary = json.loads(path.read_text(encoding='utf-8'))
    print('Código:', summary['source_commit'])
    print('Ejecución UTC:', summary['checked_at_utc'])
    print('Controles locales:', 'APROBADOS' if summary['passed'] else 'CON ERRORES')
    if not summary['passed']:
        print('Revisar all_summary.json antes de utilizar los resultados guardados.')
        return
    suites = list(ElementTree.parse(output / 'pytest.xml').getroot().iter('testsuite'))
    print('PyTest:', sum(int(s.get('tests', 0)) for s in suites), 'pruebas,',
          sum(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites), 'fallos/errores')
    data = json.loads((output / 'data_checks.json').read_text(encoding='utf-8'))
    print('Integridad:', 'APROBADA' if data['passed'] else 'CON ERRORES')
    for question, item in data['questions'].items():
        drift = item.get('drift', {})
        print(question.upper(), '| filas:', item['rows'], '| drift:',
              'ALERTA' if drift.get('failed_conditions') else 'sin alertas')
    security = json.loads((output / 'security.json').read_text(encoding='utf-8'))
    print('Repositorio:', 'APROBADO' if security['repository']['passed'] else 'CON HALLAZGOS')
    print('Bandit:', security['bandit']['findings'], 'hallazgos')
    audit_path = output / 'pip_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        affected = sum(bool(d.get('vulns')) for d in audit['dependencies'])
        print('Auditoría guardada de dependencias:', affected, 'paquetes con vulnerabilidades conocidas')
        print('Los hallazgos de dependencias siguen pendientes de tratamiento.')
    else:
        print('pip-audit: no ejecutado en esta carpeta')


def write_evidence(output):
    """Presenta los reportes guardados para consulta y captura en el navegador."""
    from contextlib import redirect_stdout
    from html import escape
    from io import StringIO

    buffer = StringIO()
    with redirect_stdout(buffer):
        show_summary(output)
    links = ''.join(f'<li><a href="{q}_drift.html">Deepchecks: drift {q.upper()}</a></li>'
                    for q in ('p1', 'p2', 'p3'))
    page = ('<!doctype html><html lang="es"><meta charset="utf-8">'
            '<title>MiBici | Calidad y seguridad</title>'
            '<style>body{font:18px system-ui;max-width:1100px;margin:36px auto;padding:20px;'
            'color:#18354a}pre{white-space:pre-wrap;background:#f2f6f8;padding:24px;'
            'font:16px/1.7 monospace}a{color:#145ca0}</style>'
            '<h1>MiBici · Calidad y seguridad</h1>'
            '<p>Resumen de resultados guardados de PyTest, Deepchecks, Bandit y pip-audit.</p>'
            f'<pre>{escape(buffer.getvalue())}</pre>'
            '<p>La aprobación de integridad y código no elimina los avisos de dependencias.</p>'
            f'<ul>{links}</ul>'
            '<p>Fuentes: pytest.xml, data_checks.json, security.json y pip_audit.json.</p>'
            '</html>')
    (output / 'evidence.html').write_text(page, encoding='utf-8')
    print('Reporte local: data/quality/evidence.html')


if __name__ == '__main__':
    raise SystemExit(main())
