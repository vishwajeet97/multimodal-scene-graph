import os
import json
import pickle as pkl
import re
from argparse import ArgumentParser

import numpy as np
from autocorrect import spell
# import difflib
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def loadGloveModel(gloveFile, isDebugging):
    # Returns the mapping as well as embedding space
    print("Loading Glove Model")

    if os.path.isfile('resources/glove/glove.42B.300d.pkl'):
        print("Pickled version found. Loading it instead...")
        with open('resources/glove/glove.42B.300d.pkl', 'rb') as f:
            return pkl.load(f)
            
    f = open(gloveFile,'r')
    model = {}
    # matrix = []
    a = np.zeros(300)
    for i, line in enumerate(tqdm(f)):
        splitLine = line.split()
        word = splitLine[0]
        if not isDebugging:
            embedding = np.array([float(val) for val in splitLine[1:]])
        else:
            embedding = a
        # matrix.append(embedding)
        model[word] = embedding
    # matrix = np.stack(matrix, axis=0)
    print("Done.",len(model)," words loaded!")

    with open('resources/glove/glove.42B.300d.pkl', 'wb+') as f:
        pkl.dump(model, f)

    return model


def mapping(glove, cfg_caption, cfg_vg):

    def _prepare_embeddings(glove, word_list):
        # Check the words and try to find a word in glove
        # label_to_id = cfg_glove.label_to_id
        # embeddings = cfg_glove.embeddings
        glove_vocab = set(glove.keys()) # Makes lookup faster
        vectors = []
        unmapped_idx = []
        for index, word in enumerate(tqdm(word_list)):
            split_words = [x for x in re.split('\.|\-|\ ', word) if x != ''] # Handles single word

            word_vector = np.zeros(300)
            count = len(split_words)
            for x in split_words: # Handles single split and zero split
                if x in glove_vocab:
                    word_vector += glove[x]
                    count -= 1
                else:
                    spell_corrected_word = spell(x) # Try one last time by spell correction
                    if spell_corrected_word in glove_vocab:
                        word_vector += glove[spell_corrected_word]
                        count -= 1
            if count == len(split_words): # All the words in split can't be mapped, shift to unmapped
                unmapped_idx.append(index)
            else:
                word_vector /= (len(split_words) - count)
                vectors.append(word_vector)

        # Partition the word list into mapped and unmapped words preserving order
        unmapped_idx = set(unmapped_idx)
        # l1 will contain the words which are mapped to glove
        l1, l2 = [], []
        l_append = (l1.append, l2.append)
        for idx, item in enumerate(word_list):
            l_append[idx in unmapped_idx](item)
        
        print('Number of words mapped to glove: %d/%d' % (len(l1), len(word_list)))
        vectors = np.stack(vectors, axis=0)
        label_to_id = {x: i for i, x in enumerate(l1)}
        id_to_label = {i: x for i, x in enumerate(l1)}

        config_embedding = {'label_to_id':label_to_id, 'id_to_label':id_to_label, 'embeddings':vectors}
        return config_embedding, set(l2)

    def _mapping(embeddings1, embeddings2):
        # Will map each vector in embeddings1 to some vector in embeddings2
        # Use KDTree for fast nearest neighbour search
        neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=4)

        neigh.fit(embeddings2)
        idx = neigh.kneighbors(embeddings1, return_distance=False)

        return [x[0] for x in idx]


    cfg_embed = {   'caption':  {
                            'objects':None,
                            'attributes':None,
                            'predicates':None},
                    'vg':       {
                            'objects':None,
                            'attributes':None,
                            'predicates':None}
                }
    unmapped_words = {'objects':None,'attributes':None,'predicates':None}

    if os.path.isfile('data/temp_embed.pkl'):
        print('Found pre-processed embeddings. Loading it instead...')
        with open('data/temp_embed.pkl', 'rb') as f, open('data/temp_unmapped.pkl', 'rb') as g:
            cfg_embed = pkl.load(f)
            unmapped_words = pkl.load(g)
    else:
        print('Preparing embeddings for caption objects...')
        cfg_embed['caption']['objects'], unmapped_words['objects'] = _prepare_embeddings(glove,\
                                                                                cfg_caption['objects'])
        print('Preparing embeddings for caption attributes...')
        cfg_embed['caption']['attributes'], unmapped_words['attributes'] = _prepare_embeddings(glove,\
                                                                                    cfg_caption['attributes'])
        print('Preparing embeddings for caption predicates...')
        cfg_embed['caption']['predicates'], unmapped_words['predicates'] = _prepare_embeddings(glove,\
                                                                                    cfg_caption['predicates'])
        
        print('Preparing embeddings for vg objects...')
        cfg_embed['vg']['objects'], _ = _prepare_embeddings(glove,\
                                                    cfg_vg['objects'])
        print('Preparing embeddings for vg attributes...')
        cfg_embed['vg']['attributes'], _ = _prepare_embeddings(glove,\
                                                        cfg_vg['attributes'])
        print('Preparing embeddings for vg predicates...')
        cfg_embed['vg']['predicates'], _ = _prepare_embeddings(glove,\
                                                        cfg_vg['predicates'])

        with open('data/temp_embed.pkl', 'wb+') as f, open('data/temp_unmapped.pkl', 'wb+') as g:
            pkl.dump(cfg_embed, f)
            pkl.dump(unmapped_words, g)
    
    # Actually mapping words from caption space to vg space
    cfg_mapping = {'objects':None,'attributes':None,'predicates':None}

    cfg_mapping['objects'] = _mapping(cfg_embed['caption']['objects']['embeddings'], cfg_embed['vg']['objects']['embeddings'])
    cfg_mapping['attributes'] = _mapping(cfg_embed['caption']['attributes']['embeddings'], cfg_embed['vg']['attributes']['embeddings'])
    cfg_mapping['predicates'] = _mapping(cfg_embed['caption']['predicates']['embeddings'], cfg_embed['vg']['predicates']['embeddings'])

    del cfg_embed['caption']['objects']['embeddings']
    del cfg_embed['caption']['attributes']['embeddings']
    del cfg_embed['caption']['predicates']['embeddings']

    del cfg_embed['vg']['objects']['embeddings']
    del cfg_embed['vg']['attributes']['embeddings']
    del cfg_embed['vg']['predicates']['embeddings']

    return cfg_embed, cfg_mapping, unmapped_words

def update_scene_graphs(scene_graph, unmapped_words):
    # remove the relationships and objects which were not mapped 
    new_scene_graph = {}
    for key, record in scene_graph.items():
        new_records = []
        for r in record:
            new_objects = [x for x in r['objects'] if x not in unmapped_words['objects']]
            new_attributes = [x for x in r['attributes'] if x['attribute'] not in unmapped_words['attributes'] and
                                                                x['object'] not in unmapped_words['objects']]
            new_relationships = [x for x in r['relationships'] if x['subject'] not in unmapped_words['objects'] and 
                                                                    x['object'] not in unmapped_words['objects'] and
                                                                    x['predicate'] not in unmapped_words['predicates']]

            new_records.append({'objects':new_objects, 'attributes':new_attributes, 'relationships':new_relationships})
        new_scene_graph[key] = new_records
    return new_scene_graph

def main(args):
    assert os.path.isfile(args.merged_scene_graphs)
    assert os.path.isfile(args.single_scene_graphs)
    
    assert os.path.isfile(args.objects_caption)
    assert os.path.isfile(args.predicates_caption)
    assert os.path.isfile(args.attributes_caption)

    assert os.path.isfile(args.objects_vg)
    assert os.path.isfile(args.predicates_vg)
    assert os.path.isfile(args.attributes_vg)

    assert os.path.isfile(args.glove_embedding)

    with open(args.merged_scene_graphs) as f, open(args.single_scene_graphs) as g:
        merged_scene_graphs = json.load(f)
        single_scene_graphs = json.load(g)

    with open(args.objects_caption) as f, open(args.predicates_caption) as g, open(args.attributes_caption) as h:
        objects_caption = json.load(f)
        predicates_caption = json.load(g)
        attributes_caption = json.load(h)

    with open(args.objects_vg) as f, open(args.predicates_vg) as g, open(args.attributes_vg) as h:
        objects_vg = json.load(f)
        predicates_vg = json.load(g)
        attributes_vg = json.load(h)

    config_caption = {'objects':objects_caption, 'attributes':attributes_caption, 'predicates':predicates_caption}
    config_vg = {'objects':objects_vg, 'attributes':attributes_vg, 'predicates':predicates_vg}

    glove = loadGloveModel(args.glove_embedding, isDebugging=args.debugging)

    config_embed, config_mapping, unmapped_words = mapping(glove, config_caption, config_vg)

    merged_scene_graphs = update_scene_graphs(merged_scene_graphs, unmapped_words)
    single_scene_graphs = update_scene_graphs(single_scene_graphs, unmapped_words)

    with open(args.merged_scene_graphs_mapped, 'w+') as f, open(args.single_scene_graphs_mapped, 'w+') as g:
        json.dump(merged_scene_graphs, f)
        json.dump(single_scene_graphs, g)

    with open('data/coco/coco_labels.pkl', 'wb+') as f, open('data/gqa/gqa_labels.pkl', 'wb+') as g:
        pkl.dump(config_embed['caption'], f)
        pkl.dump(config_embed['vg'], g)

    with open('data/mapping_coco_to_gqa.pkl', 'wb+') as f:
        pkl.dump(config_mapping, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--merged_scene_graphs', default='/home/vishwajeet/exp/IBMInternship/data/coco/merged_scene_graphs_coco.json', type=str)
    parser.add_argument('--single_scene_graphs', default='/home/vishwajeet/exp/IBMInternship/data/coco/single_scene_graphs_coco.json', type=str)
    parser.add_argument('--objects_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/objects_coco.json', type=str)
    parser.add_argument('--predicates_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/predicates_coco.json', type=str)
    parser.add_argument('--attributes_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/attributes_coco.json', type=str)

    parser.add_argument('--objects_vg', default='/home/vishwajeet/exp/IBMInternship/data/gqa/objects_gqa.json', type=str)
    parser.add_argument('--predicates_vg', default='/home/vishwajeet/exp/IBMInternship/data/gqa/predicates_gqa.json', type=str)
    parser.add_argument('--attributes_vg', default='/home/vishwajeet/exp/IBMInternship/data/gqa/attributes_gqa.json', type=str)

    parser.add_argument('--glove_embedding', default='/home/vishwajeet/exp/IBMInternship/resources/glove/glove.6B.300d.txt', type=str)

    parser.add_argument('--debugging', default=False, type=bool)

    parser.add_argument('--mapping_dict.json', default='/home/vishwajeet/exp/IBMInternship/data/coco/mapping_dict.json', type=str)
    parser.add_argument('--merged_scene_graphs_mapped', default='/home/vishwajeet/exp/IBMInternship/data/coco/merged_scene_graphs_coco_mapped.json', type=str)
    parser.add_argument('--single_scene_graphs_mapped', default='/home/vishwajeet/exp/IBMInternship/data/coco/single_scene_graphs_coco_mapped.json', type=str)
    args = parser.parse_args()

    main(args)