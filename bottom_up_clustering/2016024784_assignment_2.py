import math
import copy
from queue import PriorityQueue
from itertools import permutations

# Global variables
Coordinates = []
Pairwise_Similarities = []
Similarity_Table = []
Coord_Hash = {}
Index_Hash = {}


def output(single, complete, average, k):
    file = open('CoordinatePlane_' + str(k) + '_output.txt', 'w')
    file.write(str(k) + '\n')

    for i in range(3):
        span1, span2 = 0, 0
        cluster = []
        result = []
        content = ""

        file.write('---\n')

        if i == 0:
            result = single
            file.write('single\n')
        elif i == 1:
            result = complete
            file.write('complete\n')
        else:
            result = average
            file.write('average\n')

        for item in result:
            if item[1] == 3:
                span1 = -1 * item[0]
                cluster = item[2]
            elif item[1] < 3:
                span2 = -1 * item[0]
                break

        file.write('clusters: ')

        for j in range(len(cluster)):
            if j == (len(cluster) - 1):
                content += str(cluster[j])
            else:
                content += str(cluster[j]) + ", "

        file.write(content + '\n')
        file.write('span: ')
        file.write(str('%.8f' % span1) + ', ' + str('%.8f' % span2) + '\n')


def get_index(hash_table, coordinate):
    return hash_table[coordinate[0], coordinate[1]]


def cos_sim(v1, v2):
    dot_product = (v1[0] * v2[0]) + (v1[1] * v2[1])
    norm = math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    return dot_product / norm


def get_max(sim_table, index_set):
    _max, idx_1, idx_2 = -1, 0, 0

    for i in index_set:
        for j in index_set:
            if i != j:
                if _max < sim_table[i][j]:
                    _max = sim_table[i][j]
                    idx_1, idx_2 = i, j

    return [_max * -1, idx_1, idx_2]


def update_sim_table(n, sim_table, idx_1, idx_2, pair_sim, link_method):
    global Index_Hash

    for i in range(n):
        if i == idx_1 or i == idx_2:
            continue

        if link_method == "single":
            if sim_table[idx_1][i] < sim_table[idx_2][i]:
                sim_table[idx_1][i] = sim_table[idx_2][i]
                sim_table[i][idx_1] = sim_table[i][idx_2]

        elif link_method == "complete":
            if sim_table[idx_1][i] > sim_table[idx_2][i]:
                sim_table[idx_1][i] = sim_table[idx_2][i]
                sim_table[i][idx_1] = sim_table[i][idx_2]

        elif link_method == "average":
            sim_table[idx_1][i] = (sim_table[idx_1][i] + sim_table[idx_2][i]) / 2
            sim_table[i][idx_1] = (sim_table[i][idx_1] + sim_table[i][idx_2]) / 2
            pair_sim.append([sim_table[idx_1][i] * -1, [Index_Hash[idx_1], Index_Hash[i]]])

    if link_method == "average":
        temp = PriorityQueue()
        for i in pair_sim:
            temp.put(i)
        pair_sim.clear()
        while not temp.empty():
            pair_sim.append(temp.get())


    return [sim_table, pair_sim]


def bottom_up_clustering(n, link_method):
    global Coordinates, Pairwise_Similarities, Similarity_Table, Coord_Hash

    pair_sim = copy.deepcopy(Pairwise_Similarities)
    sim_table = copy.deepcopy(Similarity_Table)
    coord_hash = copy.deepcopy(Coord_Hash)
    cluster_set = {}
    index_set = [i for i in range(n)]
    result = []

    prio_q = PriorityQueue()

    # Traverse all pairwise similarities
    for item in pair_sim:
        cluster_list = []
        cluster_cnt = 0

        # One item of pairwise similarities
        [sim, [c_1, c_2]] = item
        idx_1, idx_2 = get_index(coord_hash, c_1), get_index(coord_hash, c_2)

        # Get max similarity in sim table
        [_max, max_idx_1, max_idx_2] = get_max(sim_table, index_set)

        # Case : v1 and v2 are in same cluster, skip
        if idx_1 == idx_2:
            continue

        # Case : item's similarity is not sim table maximum, skip
        if _max != sim:
            continue

        else:
            # Case : item's coord pair is not sim table maximum's pair, skip
            if (idx_1 != max_idx_1 or idx_2 != max_idx_2) and (idx_1 != max_idx_2 or idx_2 != max_idx_1):
                continue

        # Update sim table (pick a large one)
        [sim_table, pair_sim] = update_sim_table(n, sim_table, idx_1, idx_2, pair_sim, link_method)

        # Merge idx_2 to idx_1
        coord_hash[c_2[0], c_2[1]] = idx_1

        # If c_2 has a cluster, merge it to c_1
        if idx_2 in cluster_set:
            temp_set = cluster_set[idx_2]
            for member in temp_set:
                coord_hash[member[0], member[1]] = idx_1

        cluster_set.clear()
        index_set.clear()

        # Classify cluster
        for i in range(n):
            coord = Coordinates[i]
            idx = get_index(coord_hash, coord)

            if idx in cluster_set:
                cluster_set[idx].append(coord)
            else:
                cluster_set[idx] = [coord]
                index_set.append(idx)

        for i in cluster_set:
            temp = []
            cluster_cnt += 1
            for j in cluster_set[i]:
                temp.append(tuple(j))
            cluster_list.append(temp)

        prio_q.put([sim, cluster_cnt, cluster_list])

        if len(cluster_set) == 1:
            break

    while not prio_q.empty():
        result.append(prio_q.get())

    return result


def start():
    global Coordinates, Pairwise_Similarities, Similarity_Table, Coord_Hash, Index_Hash

    files = ['CoordinatePlane_1.txt', 'CoordinatePlane_2.txt', 'CoordinatePlane_3.txt']

    for x in files:

        # Initialize Globals
        Coordinates.clear()
        Pairwise_Similarities.clear()
        Similarity_Table.clear()
        Coord_Hash.clear()

        prio_q = PriorityQueue()
        indexing = 0

        # Open file and read meta data
        file = open(x, 'r')
        info = file.readline().replace("\n", "").split(' ')
        k, n = int(info[0]), int(info[1])

        # Get coordinates and translate coordinate to index
        while True:
            coordinate = file.readline().split('\n')[0].split(',')
            if coordinate == ['']:
                break
            Coordinates.append([int(coordinate[0]), int(coordinate[1])])
            Coord_Hash[int(coordinate[0]), int(coordinate[1])] = indexing
            Index_Hash[indexing] = [int(coordinate[0]), int(coordinate[1])]
            indexing += 1

        # 2 dim array
        Similarity_Table = [[0] * n for row in range(n)]

        # Permute coordinates to get a each similarity
        coord_permutation = list(permutations(Coordinates, 2))

        for i in range(len(coord_permutation)):
            [c_1, c_2] = coord_permutation[i]

            # Get similarity
            sim = cos_sim(c_1, c_2)

            # Get index of coordinate
            idx_1, idx_2 = get_index(Coord_Hash, c_1), get_index(Coord_Hash, c_2)

            # Push a similarity to Pairwise_Similarities, No dup (AB, BA with same sim)
            if Similarity_Table[idx_1][idx_2] == 0 and Similarity_Table[idx_2][idx_1] == 0:
                prio_q.put([-1 * sim, [c_1, c_2]])

            # Set similarity table
            Similarity_Table[idx_1][idx_2] = sim

        # Priority Queue to List
        while not prio_q.empty():
            Pairwise_Similarities.append(prio_q.get())

        # ##### Finish of global setting ##### #

        # Call clustering functions
        single = bottom_up_clustering(n, "single")
        complete = bottom_up_clustering(n, "complete")
        average = bottom_up_clustering(n, "average")

        # Make output file
        output(single, complete, average, k)


if __name__ == "__main__":
    start()
