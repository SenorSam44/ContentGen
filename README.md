# ContentGen
 pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu 

## For development (reuses local .venv)
make quick-dev

## For optimized production
make fresh-run

## Create local .venv first, then build
make optimize-venv build

## Check final image size
make size