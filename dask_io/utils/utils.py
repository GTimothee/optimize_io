import os


def flush_cache():
    os.system('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches') 


def neat_print_graph(graph):
    for k, v in graph.items():
        print("\nkey", k)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print("\nk", k2)
                print(v2, "\n")