import time

import argparse
import networkit as nk



parser = argparse.ArgumentParser(description='Process graph file.')


parser.add_argument('-f', '--graphfile', type=str, required=True,
                    help='The path to the graph file')

args = parser.parse_args()

graph_file = args.graphfile

#set number of threads
nk.setNumberOfThreads(64)

G = nk.graphio.METISGraphReader().read(graph_file)
#G = nk.graphio.METISGraphReader().read("twitter.metis")
communities = nk.community.detectCommunities(G, algo=nk.community.PLM(G, True))
nk.community.writeCommunities(communities, "communties_"+graph_file+".txt")



