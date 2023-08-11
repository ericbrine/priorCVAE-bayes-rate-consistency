#!/bin/bash
REPO_PATH="/rds/general/user/emb22/home/priorCVAE-bayes-rate-consistency"
OUT_PATH="/rds/general/user/emb22/home/output/"

# Create main script
cat > "$OUT_PATH/simulation-fullproc.pbs" <<EOF
#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=8:ompthreads=1:mem=50gb

module load anaconda3/personal
source activate bayes-rate-consistency

# Move into repository
cd $REPO_PATH

python -m bayes_rate_consistency.run project_root=$REPO_PATH output_dir=$OUT_PATH
EOF

# Execute main script
cd $OUT_PATH
qsub "covimod-fullproc.pbs"
