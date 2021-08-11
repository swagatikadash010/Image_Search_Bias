# example: ./run_all.sh ../../Collated_CSVs/Biologist.xlsx 

occupation=$1

mkdir -p runs
mkdir -p results

python prepare_ground_truth.py $occupation ./runs
python random-baseline_reranking.py $occupation ./runs
python relevance-baseline_reranking.py $occupation ./runs
python fairness-aware_reranking.py $occupation ./runs
python eval.py ./runs/ground_truth_relevance.csv ./runs/ground_truth_gender.csv ./runs

parentdir="$(dirname "$occupation")"
mv runs $parentdir
mv results/* $parentdir 

