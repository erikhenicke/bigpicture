from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from great_tables import GT


def _get_nested(mapping: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
	value: Any = mapping
	for key in dotted_key.split('.'):
		if not isinstance(value, dict):
			return default
		value = value.get(key, default)
	return value


def _extract_aliases(artifact: Any) -> set[str]:
	aliases = set()
	for alias in getattr(artifact, 'aliases', []) or []:
		if isinstance(alias, str):
			aliases.add(alias)
		else:
			alias_name = getattr(alias, 'alias', None)
			if alias_name:
				aliases.add(alias_name)
	return aliases


def _safe_name(value: str) -> str:
	return value.replace('/', '_').replace(':', '_')


def collect_best_artifacts(
	project_path: str,
	run_group: str,
	download_root: Path,
) -> pd.DataFrame:
	api = wandb.Api()
	runs = api.runs(project_path, filters={'group': run_group})

	rows: list[dict[str, Any]] = []
	seen_artifact_ids: set[str] = set()

	for run in runs:
		config = dict(run.config or {})
		for artifact in run.logged_artifacts():
			aliases = _extract_aliases(artifact)
			if 'best' not in aliases:
				continue

			artifact_id = getattr(artifact, 'id', None)
			if artifact_id and artifact_id in seen_artifact_ids:
				continue

			target_dir = download_root / f'{run.id}_{_safe_name(artifact.name)}'
			target_dir.mkdir(parents=True, exist_ok=True)
			downloaded_path = Path(artifact.download(root=str(target_dir))).resolve()

			metadata = dict(getattr(artifact, 'metadata', {}) or {})

			rows.append(
				{
					'run_id': run.id,
					'run_name': run.name,
					'run_group': run.group,
					'artifact_name': artifact.name,
					'artifact_version': getattr(artifact, 'version', None),
					'artifact_path': str(downloaded_path),
					'config.model_type': _get_nested(config, 'model_type'),
					'config.optimizer': _get_nested(config, 'optimizer'),
					'config.image_net': _get_nested(config, 'image_net'),
					'config.augmentation': _get_nested(config, 'data_augmentation', default=False),
					'val-od-worst-group-acc': metadata.get('val-od-worst-group-acc'),
					'test-od-worst-group-acc': metadata.get('test-od-worst-group-acc'),
				}
			)

			if artifact_id:
				seen_artifact_ids.add(artifact_id)

	df = pd.DataFrame(rows)
	if not df.empty:
		df = df.sort_values(
			by=['test-od-worst-group-acc', 'val-od-worst-group-acc'],
			ascending=False,
			na_position='last',
		).reset_index(drop=True)

	return df


def build_table(df: pd.DataFrame, run_group: str, output_html: Path) -> None:
	if df.empty:
		empty_df = pd.DataFrame(
			[
				{
					'config.model_type': '—',
					'config.optimizer': '—',
					'config.image_net': '—',
					'config.augmentation': False,
					'val-od-worst-group-acc': None,
					'test-od-worst-group-acc': None,
					'run_name': 'No artifacts found',
				}
			]
		)
		table_data = empty_df
	else:
		table_data = df[
			[
				'config.model_type',
				'config.optimizer',
				'config.image_net',
				'config.augmentation',
				'val-od-worst-group-acc',
				'test-od-worst-group-acc',
				'run_name',
			]
		]

	gt = (
		GT(table_data)
		.tab_header(title=f'Best artifacts for run group: {run_group}')
		.fmt_number(columns=['val-od-worst-group-acc', 'test-od-worst-group-acc'], decimals=4)
	)
	output_html.write_text(gt.as_raw_html(), encoding='utf-8')


def main() -> None:
	wandb.login()

	project_path = 'ehenicke-friedrich-schiller-universit-t-jena/fmow'
	run_group = 'densenet'

	output_csv = Path(f'results/{run_group}/best_artifacts.csv')
	output_html = Path(f'results/{run_group}/best_artifacts.html')
	download_dir = Path(f'artifacts/{run_group}')

	download_dir.mkdir(parents=True, exist_ok=True)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	output_html.parent.mkdir(parents=True, exist_ok=True)

	df = collect_best_artifacts(
		project_path=project_path,
		run_group=run_group,
		download_root=download_dir,
	)


	df.to_csv(output_csv, index=False)
	build_table(df=df, run_group=run_group, output_html=output_html)

	print(f'Collected {len(df)} best artifacts from run group "{run_group}".')
	print(f'DataFrame CSV: {output_csv}')
	print(f'Great Tables HTML: {output_html}')


if __name__ == '__main__':
	main()
