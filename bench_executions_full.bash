#### Benchmark executions for whole cloud neighbor finding, instead of a random subset
datasets=(
    "data/paris_lille/Lille_0.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
    "data/paris_lille/Lille_11.las"
    # "data/dales_las/test/5135_54435.las"
    # "data/dales_las/train/5110_54320.las"
)
radii="0.5,1.0"
high_radii="2.5"
approx_tolerances="5.0,10.0,25.0,50.0,100.0" 
number_of_threads="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
N_SEARCHES=all
KERNELS="sphere"

# Smaller executions with multiple kernels and low radii
# for dataset in "${datasets[@]}"; done
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/octree_comp_full" -r "$radii" -b "srch" -s "$N_SEARCHES" --no-warmup --repeats 1
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/point_comp_full" -r "$radii" -b "pt" -s "$N_SEARCHES" --no-warmup --repeats 1
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/algo_comp_full" -r "$radii" -b "comp" -s "$N_SEARCHES" --no-warmup --repeats 1
#     ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search_full" -r "$radii" -b "approx" -s "$N_SEARCHES" --approx-tol "$approx_tolerances" --no-warmup --repeats 1
# done

# Big executions with just the sphere kernel and high radii
for dataset in "${datasets[@]}"; do
    ./build/rule-based-classifier-cpp --kernels "$KERNELS" -i "$dataset" -o "out/octree_comp_full" -r "$high_radii" -b "srch" -s "$N_SEARCHES" --no-warmup --repeats 1
    ./build/rule-based-classifier-cpp --kernels "$KERNELS" -i "$dataset" -o "out/point_comp_full" -r "$high_radii" -b "pt" -s "$N_SEARCHES" --no-warmup --repeats 1
    ./build/rule-based-classifier-cpp --kernels "$KERNELS" -i "$dataset" -o "out/algo_comp_full" -r "$high_radii" -b "comp" -s "$N_SEARCHES" --no-warmup --repeats 1
    ./build/rule-based-classifier-cpp --kernels "$KERNELS" -i "$dataset" -o "out/approx_search_full" -r "$high_radii" -b "approx" -s "$N_SEARCHES" --approx-tol "$approx_tolerances" --no-warmup --repeats 1
done