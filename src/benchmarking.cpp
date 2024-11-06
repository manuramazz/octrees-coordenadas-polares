#include "octree.hpp"
#include "octree_v2.hpp"
#include "octree_linear.hpp"
#include "octree_pointer.hpp"
#include "benchmarking.hpp"


void benchmarking::runBenchmark(const std::vector<Lpoint>& points, Octree& originalOctree, PointerOctree& pointerOctree, 
    LinearOctree& linearOctree) {
    size_t repeats = 10, neigh_searches = 100;
    std::vector<int> random_ints;
    std::vector<float> random_radius;
    for (int i = 0; i < neigh_searches; i++) {
      random_ints.push_back(rand() % points.size());
      random_radius.push_back((rand() % 100) * 0.2f);
    }

    auto neighbour_benchmark = [&](auto& octree, const std::string& name) {
      benchmarking::benchmark(name, repeats, [&]() {
          for (int i = 0; i < neigh_searches; i++) {
              auto neigh = octree.searchNeighbors(points[random_ints[i]], random_radius[i]);
          }
      });
    };

  neighbour_benchmark(pointerOctree, "searchNeighbors en octree de punteros");
  neighbour_benchmark(originalOctree, "searchNeighbors en octree original");
  neighbour_benchmark(linearOctree, "searchNeighbors en octree lineal");
}

