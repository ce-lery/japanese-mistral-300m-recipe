# simple example

## Overview

This is a simple pretraining example.

## Getting Started

create environment

```bash
uv sync --extra build
uv sync --extra build --extra compile
```

download japanese wiki dataset.

```bash
bash dataset.sh
```

pretrain 300m mistral.

```bash
bash train.sh
```

pretrain 2b mistral.

```bash
bash train_2b.sh
```

