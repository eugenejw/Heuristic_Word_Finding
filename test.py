import networkx as nx


class Test(object):
    def __init__(self):
        self._meaningful_words = [((0, 4), 'face'), ((0, 8), 'facebook'), ((1, 4), 'ace'), ((4, 6), 'bo'), ((4, 7), 'boo'), ((4, 8), 'book'), ((6, 9), 'ok'),((8,10), 'in'), ((8, 11), 'king'), ((10, 14), 'girl')]
#        self._init_graph(self._meaningful_words)
        pass

    def _init_graph(self, meaningful_words):
        #[((0, 3),"face"), ((0, 7),"facebook"),((1, 3),"ace"), ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok"), ((8, 19),"anotherword")]
        G = nx.Graph()
        G.add_nodes_from(meaningful_words)
        for each in meaningful_words:
            for each_2 in meaningful_words:
                if each == each_2:
                    continue
                elif self._intersect(each[0], each_2[0]):
                    if (each[0], each_2[0]) in G.edges():
                        continue
                    else:
                        G.add_edge(each, each_2)
        return G

    def _intersect(self, tuple_0, tuple_1):
        '''
        finds intersection of two words
        '''
        x = range(tuple_0[0],tuple_0[1])
        y = range(tuple_1[0],tuple_1[1])
        xs = set(x)
        return xs.intersection(y)

    def _find_components(self):
        meaningful_words = self._meaningful_words
        G = nx.Graph()
        G = self._init_graph(meaningful_words)
        components = []
        components = nx.connected_component_subgraphs(G)
        return components
    

    def opt(self):
        components = self._find_components()
#        print "{}components found\n".format(len(list(components)))
        for component in components:
            print "hererer\n"
            if len(component.nodes()) == 1:
               #means the component contains only one word
               print "[you should not encounter this]single compond: {}\n".format(component.nodes())
               return True

            #return a list of optimal words in format of (pos, "word")
    
            nodes = component.nodes() #initially nodes = all nodes in component
            nodes.sort()
            print "nodes are {}\n".format(nodes)

            def search(component, nodes=nodes, node=nodes[0], flag='init'):
                if not nx.non_neighbors(component, node) and flag != 'init':
                    print "no neighbor returned: {}".format([node])
                    return node
                elif nx.non_neighbors(component, node) and flag != 'init':
                        #only look at word forwad
                    flag = "HASNOT"
                    for each in nx.non_neighbors(component, node):

                        if each[0][0] > node[0][0]:
                            #if non_neighbor has forward neighbors
                            #keep the flag
                            flag = "HAS"
                            break
                        else:
                            #all non_neighbors are in front of the node
                            pass

                    if flag == "HASNOT":
                        print "no forward neighbor returned: {}".format([node])
                        return node
                    else:
                        #means it has non_neighbor following it
                        pass
                
                def candidates():
                    for node in nodes:
                            print "node is {}\n".format(node)
                            #node, say, ((0, 3),"face")

                            if list(nx.non_neighbors(component, node)) != []:
                             for each_non_neighbor in nx.non_neighbors(component, node):
                                candidate_nodes = [node]

                                if each_non_neighbor[0][0] > node[0][0]:
                                    print "each_non_neighbor is {}\n".format(each_non_neighbor)
                                #boo is one of the non_neighbors of "face"
                                    candidate_nodes.append(search(component, [each_non_neighbor], each_non_neighbor, flag=''))
                                #booking, no further non_neighbors, so will return a node itself
                                #boo, has two non_neighbors -- "king" and "girl", will return the 
                                yield candidate_nodes
                                #yield (score, [list of nodes])
                            else:
                                print "HHHHEEEERRRREEEE\n"
                                candidate_nodes = [node]
                                yield candidate_nodes

                return list(candidates())

            optimized_words = search(component) #in format of (pos, "word1xxword2..")
            print "optimized_words is {}\n".format(optimized_words)
            s = []
            for i in optimized_words:
                if i not in s:
                    s.append(i)
            print "s is {}".format(s)

            for each in s:
                print "\n"
                
                print each

            return s
            #return optimized_words
        
    #optimalized_components.append(self.opt())
        
    #return optimalized_components  
