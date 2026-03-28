"""
Submit three parallel SageMaker Training Jobs (copula, ctgan, tvae),
then poll until all complete and download synthetic outputs locally.

Usage:
  python sagemaker/launch_jobs.py
"""

from __future__ import annotations

import time
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ACCOUNT    = '590184025853'
REGION     = 'us-east-1'
BUCKET     = f'synthetic-ml-{ACCOUNT}'
ROLE_ARN   = f'arn:aws:iam::{ACCOUNT}:role/SyntheticMLSageMakerRole'
DATA_S3    = f's3://{BUCKET}/synthea/data'
OUTPUT_S3  = f's3://{BUCKET}/synthea/output'

# ml.m5.4xlarge: 16 vCPU, 64 GB RAM — good for tabular; no GPU needed
INSTANCE_TYPE = 'ml.m5.4xlarge'

GENERATORS = [
    {'name': 'copula', 'epochs': 1,   'batch_size': 500},  # copula ignores epochs
    {'name': 'ctgan',  'epochs': 300, 'batch_size': 500},
    {'name': 'tvae',   'epochs': 300, 'batch_size': 500},
]


def submit_jobs() -> dict[str, str]:
    """Submit all three training jobs and return {generator: actual_job_name}."""
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION),
        default_bucket=BUCKET,
    )

    job_names = {}
    for gen in GENERATORS:
        name = gen['name']
        base_job_name = f'synth-{name}'

        estimator = PyTorch(
            entry_point='train.py',
            source_dir='sagemaker',
            role=ROLE_ARN,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version='2.2',
            py_version='py310',
            base_job_name=base_job_name,
            hyperparameters={
                'generator':    name,
                'target-col':   'DECEASED',
                'epochs':       gen['epochs'],
                'batch-size':   gen['batch_size'],
                'random-state': 42,
            },
            output_path=OUTPUT_S3,
            sagemaker_session=session,
        )

        estimator.fit(
            inputs={'train': DATA_S3},
            wait=False,  # non-blocking — all 3 run in parallel
            logs=False,
        )
        # Capture the actual job name assigned by SageMaker
        actual_job_name = estimator.latest_training_job.job_name
        job_names[name] = actual_job_name
        print(f'Submitted {name} -> {actual_job_name}')

    return job_names


def poll_jobs(job_names: dict[str, str], poll_interval: int = 30) -> dict[str, str]:
    """Poll all jobs until they finish. Returns {generator: final_status}."""
    sm = boto3.client('sagemaker', region_name=REGION)
    pending = dict(job_names)
    statuses: dict[str, str] = {}

    print(f'\nPolling {len(pending)} jobs every {poll_interval}s ...')
    while pending:
        time.sleep(poll_interval)
        for gen, job in list(pending.items()):
            resp = sm.describe_training_job(TrainingJobName=job)
            status = resp['TrainingJobStatus']
            if status in ('Completed', 'Failed', 'Stopped'):
                statuses[gen] = status
                duration = resp.get('TrainingTimeInSeconds', '?')
                print(f'  [{status}] {gen} ({job})  {duration}s')
                del pending[gen]
            else:
                phase = resp.get('SecondaryStatus', status)
                print(f'  [{phase}] {gen} ...')

    return statuses


def download_outputs(job_names: dict[str, str], local_dir: str = 'data/synthetic') -> None:
    """Download model.tar.gz for each completed job and extract synthetic CSV."""
    import tarfile

    s3 = boto3.client('s3', region_name=REGION)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for gen, job in job_names.items():
        s3_key = f'synthea/output/{job}/output/model.tar.gz'
        local_tar = local_dir / f'{gen}_model.tar.gz'
        extract_dir = local_dir / f'{gen}_output'
        extract_dir.mkdir(exist_ok=True)

        print(f'Downloading s3://{BUCKET}/{s3_key} ...')
        try:
            s3.download_file(BUCKET, s3_key, str(local_tar))
            with tarfile.open(local_tar) as tar:
                tar.extractall(extract_dir)
            # Move synthetic CSV to data/synthetic/
            synth_csv = extract_dir / f'{gen}_synthetic.csv'
            if synth_csv.exists():
                dest = local_dir / f'{gen}_synthetic.csv'
                synth_csv.rename(dest)
                print(f'  -> {dest}')
            else:
                print(f'  Warning: {synth_csv} not found in archive')
        except Exception as e:
            print(f'  Error downloading {gen}: {e}')


if __name__ == '__main__':
    print('=== Submitting SageMaker Training Jobs ===')
    job_names = submit_jobs()

    print('\n=== Job names ===')
    for gen, job in job_names.items():
        print(f'  {gen}: {job}')

    # Save job names so you can re-poll if needed
    import json
    with open('sagemaker/job_names.json', 'w') as f:
        json.dump(job_names, f, indent=2)
    print('\nSaved to sagemaker/job_names.json')

    print('\n=== Polling for completion ===')
    statuses = poll_jobs(job_names)

    print('\n=== Final statuses ===')
    for gen, status in statuses.items():
        print(f'  {gen}: {status}')

    completed = [g for g, s in statuses.items() if s == 'Completed']
    if completed:
        print(f'\n=== Downloading outputs for: {completed} ===')
        download_outputs({g: job_names[g] for g in completed})
        print('\nDone. Synthetic CSVs in data/synthetic/')
    else:
        print('\nNo jobs completed successfully.')
