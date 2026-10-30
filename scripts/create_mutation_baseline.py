"""Explicitly snapshot validated reports measured by a clean-checkout audit run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from compare_mutation_baselines import SCOPES, fingerprint, read, require, validate_fingerprint, validate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=SCOPES, required=True)
    for name in ('covered', 'inclusive', 'repository-root', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.repository_root.resolve()
        git = lambda *a: subprocess.check_output(['git', '-C', str(root), *a], text=True).strip()
        require(not git('status', '--porcelain'), 'Baseline generation requires a clean checkout')
        manifest = read(args.manifest)
        require(manifest['revision'] == git('rev-parse', 'HEAD') and manifest['source_clean'] is True, 'Manifest does not attest a clean current revision')
        record = manifest['scopes'][args.scope]
        require(record['covered_returncode'] == record['inclusive_returncode'] == 0, 'Audit target failed')
        from compare_mutation_baselines import sha
        require(record['covered_sha256'] == sha(args.covered) and record['inclusive_sha256'] == sha(args.inclusive), 'Reports differ from measured manifest')
        fp = fingerprint(root, args.scope, manifest['execution'])
        validate_fingerprint(fp, args.scope)
        require(record['fingerprint'] == fp, 'Fingerprint changed since measurement')
        mutants = validate_pair(read(args.covered), read(args.inclusive), args.scope, root)
        result = {'schema_version': 1, 'scope': args.scope, 'baseline_revision': manifest['revision'], 'fingerprint': fp, 'mutants': list(mutants.values())}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f'Baseline creation failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
