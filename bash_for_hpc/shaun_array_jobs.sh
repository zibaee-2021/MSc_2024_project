# SGE script for running array jobs

#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -l h_rt=24:00:00
#$ -l scratch0free=6G
#$ -l tscratch=6G

#$ -l hostname="!(walter)"
#$ -l tmem=12G
#$ -l gpu=true
#$ -R y
#$ -P ted

#$ -o /dev/null
#$ -e /dev/null

#$ -t 1-6902:10
#$ -N fc_tedr

conda activate /SAN/orengolab/af_esm/tools/smk_conda_merizo_env #pytorch

for ((i=0; i<SGE_TASK_STEPSIZE; i++))
do
        ((curr=SGE_TASK_ID+i))
        cmd=$(sed -n ${curr}p $1) #$(getline $curr $1)
        eval $cmd
done

#cmd=$(sed -n ${SGE_TASK_ID}p $1)
#eval $cmd