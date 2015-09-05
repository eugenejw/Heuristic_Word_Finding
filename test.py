import networkx as nx


class Test(object):
    def __init__(self):
        self._meaningful_words = [((0, 2), 'fa'), ((0, 4), 'face'), ((1, 3), 'ac'), ((1, 4), 'ace'), ((2, 4), 'ce'), ((3, 5), 'eb'), ((4, 6), 'bo'), ((4, 7), 'boo'), ((4, 8), 'book')]
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
        components = list(nx.connected_component_subgraphs(G))
        return components

    def opt(self):
        components = self._find_components()
        optimalized_components = []
        for component in components:
            if len(component.nodes()) == 1:
               #means the component contains only one word
               optimalized_components.append((component.nodes()[0][0][0], component.nodes()[0][0][1])) #("pos, "word")
               continue

            #return a list of optimal words in format of (pos, "word")
    
            nodes = component.nodes() #initially nodes = all nodes in component

            def search(component, nodes=[], flag='init'):
                if not nx.non_neighbors(component, node) and node != 'init':
                    return [node]
                elif nx.non_neighbors(component, node) and node != 'init':
                        #only look at word forwad
                    flag == True
                    for each in nx.non_neighbors(component, node):
                        if each[0][0] > node[0][0]:
                            pass
                        else:
                            flag = False

                        if flag == True:
                            return [node]
#                        return (component.nodes()[0][0][0], component.nodes()[0][0][1]) #("pos, "word")
                
                def candidates():
                        for node in nodes.sort():
                            #node, say, ((0, 3),"face")
                            candidate_nodes = [node]
                            for each_non_neighbor in non_neighbors(component, node):

                                #boo is one of the non_neighbors of "face"
                                candidate_nodes.append(search(component, each_non_neighbor, each_non_neighbor))
                                #booking, no further non_neighbors, so will return a node itself
                                #boo, has two non_neighbors -- "king" and "girl", will return the 
                                
                                yield candidate_nodes
                                #yield (score, [list of nodes])

                return list(candidates())

            optimized_words = search(component) #in format of (pos, "word1xxword2..")
            print optimized_words
            return optimized_words

        optimalized_components.append(get_optimal_words(self,component))
        
        return optimalized_components  
