datasets=(
    "data/paris_lille/Lille_0.las"
    "data/paris_lille/Paris_Luxembourg_6.las"
    "data/paris_lille/Lille_11.las"
    "data/dales_las/test/5135_54435.las"
    "data/dales_las/train/5110_54320.las"
)
radii="0.5,1.0,2.5,5.0"

datasets_high_density=(
    "data/speulderbos/Speulderbos_2017_TLS.las"
    "data/semantic3d/bildstein_station1_xyz_intensity_rgb.txt" 
    "data/semantic3d/sg27_station8_intensity_rgb.txt"
)

radii_high_density="0.05,0.1,0.25,0.5"
approx_tolerances="5.0,10.0,25.0,50.0,100.0" 
number_of_threads="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40"
N_SEARCHES=5000

for dataset in "${datasets[@]}"; do
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/octree_comp" -r "$radii" -b "srch" -s "$N_SEARCHES"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/octree_comp" -r "$radii_high_density" -b "srch" -s "$N_SEARCHES"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/point_comp" -r "$radii" -b "pt" -s "$N_SEARCHES"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/point_comp" -r "$radii_high_density" -b "pt" -s "$N_SEARCHES"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/algo_comp" -r "$radii" -b "comp" -s "$N_SEARCHES"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/parallel_full" -r "0.5,1.0,2.0,3.0,4.0" -b "parallel" -s "$N_SEARCHES" --num-threads "$number_of_threads"
done

for dataset in "${datasets_high_density[@]}"; do
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/algo_comp" -r "$radii_high_density" -b "comp" -s "$N_SEARCHES"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "$radii" -b "approx" -s "$N_SEARCHES" --approx-tol "$approx_tolerances"
    ./build/rule-based-classifier-cpp -i "$dataset" -o "out/approx_search" -r "$radii_high_density" -b "approx" -s "$N_SEARCHES" --approx-tol "$approx_tolerances"
    # too slow
    # ./build/rule-based-classifier-cpp -i "$dataset" -o "out/parallel" -r "$radii_high_density" -b "parallel" -s "$N_SEARCHES" --num-threads "$number_of_threads"
done
