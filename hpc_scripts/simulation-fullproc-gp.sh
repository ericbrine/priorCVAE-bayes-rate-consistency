#!/bin/bash
REPO_PATH="/rds/general/user/emb22/home/priorCVAE-bayes-rate-consistency"
OUT_PATH="/rds/general/user/emb22/home/output"

# Create main script
cat > "$OUT_PATH/simulation-fullproc-gp.pbs" <<EOF
#!/bin/bash
#PBS -l walltime=16:00:00
#PBS -l select=1:ncpus=32:ompthreads=1:mem=50gb

module load anaconda3/personal
source activate bayes

# Move into repository
cd $REPO_PATH
python -m bayes_rate_consistency.run project_root=$REPO_PATH output_root=$OUT_PATH model=GP-AA dataset.size=1000 dataset.intensity=inCOVID
python -m bayes_rate_consistency.run project_root=$REPO_PATH output_root=$OUT_PATH model=GP-AA dataset.size=1000 dataset.intensity=preCOVID

python -m bayes_rate_consistency.run project_root=$REPO_PATH output_root=$OUT_PATH model=GP-AA dataset.size=2000 dataset.intensity=inCOVID
python -m bayes_rate_consistency.run project_root=$REPO_PATH output_root=$OUT_PATH model=GP-AA dataset.size=2000 dataset.intensity=preCOVID

python -m bayes_rate_consistency.run project_root=$REPO_PATH output_root=$OUT_PATH model=GP-AA dataset.size=5000 dataset.intensity=inCOVID
python -m bayes_rate_consistency.run project_root=$REPO_PATH output_root=$OUT_PATH model=GP-AA dataset.size=5000 dataset.intensity=preCOVID
EOF

# Execute main script
cd $OUT_PATH
qsub "simulation-fullproc-gp.pbs"
