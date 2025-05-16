## Overview

This repository contains the code for training a document-summarizing agent using reinforcement learning. The agent is based on [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) and learns from its own experience through the [ART](https://github.com/openpipe/art) reinforcement learning framework. While training runs are non-deterministic, the agent usually outperforms gpt-4o within 5-20 training steps, which equates to less than 20 minutes of training time on a single H100 GPU.

<img src="assets/benchmarks/summarizer/accuracy-training-progress.svg" alt="Benchmark Win Rate Comparison" width="500"/>

## Getting Started

### 1. Install dependencies

If you haven't already, install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Then install the project dependencies by running `uv sync`.

### 2. Install SkyPilot/RunPod

We'll be using `SkyPilotBackend` to manage the GPU that your model will be trained on. In order for the backend to work, you'll need to have SkyPilot installed on your machine and provide it with the credentials to spin up machines on at least one infra provider.

We recommend using RunPod because of their ease of use, but any infra provider that SkyPilot [supports](https://docs.skypilot.co/en/latest/overview.html#bringing-your-infra) will work.

Follow RunPod's **Getting Started** guide [here](https://docs.runpod.io/integrations/skypilot/). You'll have to provide a credit card to use RunPod, but you'll only pay for the time your GPUs are running.

### 3. Set up optional environment variables found in `.env.example`.

In a new `.env` file at the root of the repository, set the following optional environment variables:

- `WANDB_API_KEY` - Enables metric logging to Weights & Biases.
- `OPENPIPE_API_KEY` - Enables chat completion logging to OpenPipe.
- `OPENAI_API_KEY` - Will be necessary for later comparison benchmarks, but not used for training.

To enable model and logging backup to S3, you'll also need to provide AWS credentials. These are necessary for generating the benchmarks found in the `benchmarks` directory, but not for training itself. If you don't already have AWS credentials with create/read/write permissions for s3 buckets, follow the instructions [here](CONFIGURING_AWS.md).

- `AWS_ACCESS_KEY_ID` - Your AWS access key ID, which should have create/read/write permissions for s3 buckets.
- `AWS_SECRET_ACCESS_KEY` - Your matching secret access key.
- `AWS_REGION` - The region of the S3 bucket.
- `BACKUP_BUCKET` - The name of the S3 bucket in which to store model checkpoints and logging data. Can be a new bucket or an existing one.

### 4. Run the training script

```bash
uv run python src/summarizer/train.py
```

The following steps execute when a training run on a new cluster begins:

- **Spin up a cluster with 1 H100 GPU.**
  - This usually takes about 10 minutes, but RunPod occasionally has network throughput issues that can cause the cluster to take up to 30 minutes to spin up. Once the cluster is provisioned, it can be used for subsequent training runs without going through this process again.
- **Register the model with ART.**
  - This usually takes less than 5 minutes, though it can require up to 30 minutes if RunPod experiences network issues.
- **Download the model checkpoint from S3.**
  - Usually takes a few seconds.
- **Train the model for a specified number of steps.**
  - Training itself should be pretty quick (each step takes less than a minute), but the total training time will depend on how many steps you run for. During training, the model checkpoint is saved to S3 after each step.
- **Upload the final model checkpoint to S3.**
  - This usually takes a few seconds.

### 5. Shutting down the cluster

When you're done training and running benchmarks, you can shut down the cluster in two ways:

Through the CLI:

```bash
uv run sky down <cluster-name>
```

or through code:

```python
DESTROY_AFTER_RUN = True

if DESTROY_AFTER_RUN:
    await backend.down()
```

However, since spinning up clusters is a time-intensive process, we recommend keeping clusters alive until you're sure you won't be using them in the near future.

### Running Benchmarks

The `benchmark_models.py` script will compare the performance of the trained model to `gpt-4o`, `gpt-4o-mini`, and `gpt-4.1`.

Before running the benchmark script, make sure you've provided a valid `OPENAI_API_KEY` and the AWS credentials detailed in step 3. These credentials are necessary for the script to upload the benchmark results to S3.

```bash
uv run python benchmarks/benchmark_models.py
```

This script will:

- Run each benchmarked model through 48 games of Tic Tac Toe.
- Record the proportion of games won by each model.
- Upload the results to S3.

Once the benchmark generation script has finished running, you can view the results and generate visual charts by navigating to `benchmarks/display_benchmarks.ipynb` and running the cells. After running all the cells, you should see something like the following:

<img src="assets/benchmarks/summarizer/accuracy-training-progress.svg" alt="Benchmark Win Rate Comparison" width="500"/>

_The win rate of the trained model compared to gpt-4o, gpt-4o-mini, and gpt-4.1 at each training step. By step 5 of this training run, the trained model outperforms every other model._

<img src="assets/benchmarks/summarizer/accuracy-comparison.svg" alt="Benchmark Accuracy Comparison" width="500"/>

_A side-by-side comparison of the win rates of the trained model, gpt-4o, gpt-4o-mini, and gpt-4.1. The trained model began with a 0% win rate, but by the final step, it had a 96% win rate._

**Notes on win rate calculation:**

- The win rate is calculated as the average "win" metric of each model's performance over the course of 48 games.
- For each game, the "win" metric is 1 if the model wins, 0.5 if it draws, and 0 if it loses.
