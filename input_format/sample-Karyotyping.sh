#!/bin/bash
#SBATCH --job-name=Karyotyping # Job name
#SBATCH --mem=8G               # Job memory request
#SBATCH --time=2:00:00         # Time limit days-hrs:min:sec
#SBATCH --partition normal         # Submit to the week-long queue
#SBATCH --output=%x.%j.out      # Standard output log
#SBATCH --error=%x.%j.err       # Standard error log
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1


module load devel java/21.0.4

nextflow run \
	Karyotype_Main/main.nf \
	-with-conda \
	--resume \
	--sampleSheet samples.txt \
	--profile conda \
	--bamFiles bams.txt



# if only replot
# /home/users/luqiao/.conda/envs/Karyotyping/bin/nextflow run \
#         Karyotype_Main/main.nf \
#     --base 2 \
#     --y_log false \
#     --plot_inputs plot_inputs.tsv