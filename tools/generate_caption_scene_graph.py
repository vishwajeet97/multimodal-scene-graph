import os
import json
import sys
import subprocess
import argparse
import itertools
import collections

from tqdm import tqdm

def preprocess(coco_caption, args):
    import spacy
    from autocorrect import spell

    def custom_pipeline_function(doc):
        # Removes the punctuation and implodes words with hypen
        tokens = []
        for token in doc:
            if token.lemma_ != '-PRON-' and not token.is_punct:
                tokens.append("".join((spell(token.lemma_),token.whitespace_)))
            elif token.lemma_ == '-PRON-':
                tokens.append("".join((token.lower_,token.whitespace_)))
        return "".join(tokens).strip()

    raw_sentences = []
    for record in coco_caption:
        for sent in record['sentences']:
            raw_sentences.append(sent['raw'])

    # Do the processing in batched manner
    nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp.add_pipe(custom_pipeline_function, 'custom_component', last=True)

    processed_sentences = [text for text in tqdm(nlp.pipe(raw_sentences, batch_size=50))]
    for x in processed_sentences[:10]:
        print(x)
    # for doc in nlp.pipe(raw_sentences, batch_size=50, n_threads=4):
        # processed_sentences.append(" ".join(token.text for token in doc if token.text not in ))

    index = 0
    for record in coco_caption:
        for sent in record['sentences']:
            sent['raw'] = processed_sentences[index]
            index += 1

def run_spice(coco_caption, args):
    # Run first code to make merged scene-graphs
    # will require to split the list as spice code gives OOM
    def split_list(alist, wanted_parts=1):
            length = len(alist)
            return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                     for i in range(wanted_parts) ]

    def run_spice_on_split(split_data):
        spiced_data = []
        for i, split_instance in enumerate(tqdm(split_data)):
            temp_file_name = 'temp_input' + '.' + str(i)
            temp_out_file_name = 'temp_output' + '.' + str(i)
            with open(temp_file_name, 'w+') as f:
                json.dump(split_instance, f)

            # Run spice on parts of the data
            cmd = 'java -Xmx8G -jar %s %s -out %s -detailed -threads 4' % (args.spice_jar, temp_file_name, temp_out_file_name)

            return_code = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

            with open(temp_out_file_name, 'r') as f:
                spiced_data.append(json.load(f))

            os.remove(temp_file_name)
            os.remove(temp_out_file_name)

        return spiced_data
        
    split_data = split_list(coco_caption, wanted_parts=args.split)
    print("Running SPICE for merged scene graphs....")
    merged_scene_graphs = itertools.chain.from_iterable(run_spice_on_split(split_data))
    
    split_data = []
    # Slicing across captions, i.e. taking a single caption for each id
    # Passing the sliced data to spice and rejoining afterwards
    for i in range(5): # Change this to reflect the number of ref captions
        single_caption_data = []
        for record in coco_caption:
            new_record = {"image_id":record["image_id"], "test":"dummy sentence", "refs":[record["refs"][i]]}
            single_caption_data.append(new_record)

        split_data.append(single_caption_data)

    print("Running SPICE for single scene graphs....")
    spiced_single_data = run_spice_on_split(split_data)
    single_scene_graphs = []

    for records in zip(*spiced_single_data):
        ref_tuples = [x["ref_tuples"] for x in records]
        single_scene_graphs.append({"image_id":records[0]["image_id"],"test_tuples":None,"ref_tuples":ref_tuples})


    return merged_scene_graphs, single_scene_graphs

def postprocess(scene_graph, merged=True):
    # The output file produced by SPICE gives out 
    # reference tuples containing either just objects,
    # objects alongwith their attributes or relationships

    Config = collections.namedtuple('Config', ['scene_graph','objects','attributes','predicates'])
    new_scene_graph = {}
    objects = set()
    attributes = set()
    predicates = set()
    for i, record in enumerate(scene_graph):
        # Make the ref_tuples of merged into a list to reuse the entire code
        ref_tuples_list = [record['ref_tuples']] if merged else record['ref_tuples']
        
        for ref_tuples in ref_tuples_list:
            scene_record = {'objects':set(),'attributes':[],'relationships':[]}
            for ref_tuple in ref_tuples:
                # It spits out some noun with a close noun using N/N. Just 
                # pick up the second one. 
                if len(ref_tuple['tuple']) == 1: # Only objects
                    obj = ref_tuple['tuple'][0].split('/')[-1]
                    scene_record['objects'].add(obj)
                    objects.add(obj)
                elif len(ref_tuple['tuple']) == 2: # Objects alongwith their attributes
                    obj = ref_tuple['tuple'][0].split('/')[-1]
                    attr = ref_tuple['tuple'][1]
                    scene_record['attributes'].append({'object':obj, 'attribute':attr})
                    attributes.add(attr)
                    objects.add(obj)
                elif len(ref_tuple['tuple']) == 3: # Relationships
                    obj = ref_tuple['tuple'][2].split('/')[-1]
                    subj = ref_tuple['tuple'][0].split('/')[-1]
                    pred = ref_tuple['tuple'][1]
                    scene_record['relationships'].append({'object':obj, 'predicate':pred, 'subject':subj})
                    predicates.add(pred)
                    objects.add(obj)
                    objects.add(subj)

            scene_record['objects'] = sorted(list(scene_record['objects']))
            new_scene_graph[record['image_id']] = new_scene_graph.get(record['image_id'],[]) + [scene_record]
    
    cfg = Config(new_scene_graph, sorted(list(objects)), sorted(list(attributes)), sorted(list(predicates)))

    return cfg

def main(args):
    # number of splits should be greater than 1
    assert args.split > 0
    assert os.path.isfile(args.spice_jar)
    assert os.path.isfile(args.coco_caption)


    if args.preprocess:
        with open(args.coco_caption, 'r') as f:
            coco_caption = json.load(f)['images']

        if args.debugging:
            coco_caption = coco_caption[:20]

        preprocess(coco_caption, args)

        with open(args.coco_caption_processed, 'w+') as f:
            json.dump(coco_caption, f)

        sys.exit('Preprocessing done...')
    
    with open(args.coco_caption) as f:
        coco_caption = json.load(f)['images']

    if args.debugging:
        coco_caption = coco_caption[:20]    

    coco_caption_for_spice = []

    for record in coco_caption:
        coco_id = record['cocoid']
        coco_caption_for_spice.append({"image_id": coco_id, "test":"dummy sentence", "refs":[]})
        for sent in record['sentences']:
            coco_caption_for_spice[-1]["refs"].append(sent['raw'])

    merged_scene_graphs, single_scene_graphs = run_spice(coco_caption_for_spice, args)

    merged_scene_graphs, single_scene_graphs = postprocess(merged_scene_graphs, merged=True), postprocess(single_scene_graphs, merged=False)

    objects = sorted(set.union(set(merged_scene_graphs.objects), set(single_scene_graphs.objects)))
    attributes = sorted(set.union(set(merged_scene_graphs.attributes), set(single_scene_graphs.attributes)))
    predicates = sorted(set.union(set(merged_scene_graphs.predicates), set(single_scene_graphs.predicates)))

    with open(args.merged_scene_graphs, 'w+') as f, open(args.single_scene_graphs, 'w+') as g:
        json.dump(merged_scene_graphs.scene_graph, f)
        json.dump(single_scene_graphs.scene_graph, g)

    with open(args.objects_caption, 'w+') as f, open(args.attributes_caption, 'w+') as g, open(args.predicates_caption, 'w+') as h:
        json.dump(objects,f)
        json.dump(attributes,g)
        json.dump(predicates,h)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--spice_jar', default='/home/vishwajeet/exp/IBMInternship/lib/spice/spice-1.0.jar', type=str)
    parser.add_argument('--coco_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/dataset_coco.json', type=str)
    # parser.add_argument('--rel_annotations_caption_individual', default='/home/vishwajeet/exp/IBMInternship/data/coco/rel_annotations_caption_individual.json', type=str)
    # parser.add_argument('--rel_annotations_caption_merged', default='/home/vishwajeet/exp/IBMInternship/data/coco/rel_annotations_caption_merged.json', type=str)
    # parser.add_argument('--detections_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/detections_caption.json', type=str)
    parser.add_argument('--merged_scene_graphs', default='/home/vishwajeet/exp/IBMInternship/data/coco/merged_scene_graphs_coco_wo_spell_check.json', type=str)
    parser.add_argument('--single_scene_graphs', default='/home/vishwajeet/exp/IBMInternship/data/coco/single_scene_graphs_coco_wo_spell_check.json', type=str)
    parser.add_argument('--objects_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/objects_coco_wo_spell_check.json', type=str)
    parser.add_argument('--predicates_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/predicates_coco_wo_spell_check.json', type=str)
    parser.add_argument('--attributes_caption', default='/home/vishwajeet/exp/IBMInternship/data/coco/attributes_coco_wo_spell_check.json', type=str)

    parser.add_argument('--split', default=5, type=int, help='entire coco captions results in OOM in spice')
    parser.add_argument('--debugging', default=False, type=bool, help='debugging flag only loads 100 images')
    parser.add_argument('--preprocess', default=False, type=bool, help='preprocessing of coco captions')
    parser.add_argument('--coco_caption_processed', default='/home/vishwajeet/exp/IBMInternship/data/coco/dataset_coco_processed.json', type=str)

    args = parser.parse_args()
    main(args)