from data_process.sampling import EpisodeDescriptionSampler
from data_process.dataset_spec import DatasetSpecification
from data_process.config import EpisodeDescriptionConfig
from collections import defaultdict
import gin
import json
from data_process.learning_spec import Split
import numpy as np

def get_data(path):
    result = []
    with open(path, 'r') as src:
        for line in src:
            line = json.loads(line)
            result.append(line)
    return result

def get_classes_infomation(vocabPath, filePath):
    classes = []
    examples = defaultdict(list)
    class_names_to_ids = {}
    with open(vocabPath, 'r') as src:
        for line in src:
            classes.append(line.split('\n')[0])
    datas = get_data(filePath)
    for line in datas:
        examples[line["intent"]].append(line)
    class_names = {}
    examples_per_class = {}
    for i in range(len(classes)):
        class_names[i] = classes[i]
        examples_per_class[i] = len(examples[classes[i]])
    
    if(len(classes) != len(examples.keys())):
        print("Wrong vocab")
    for key in class_names.keys():
        class_names_to_ids[class_names[key]] = key
    
    return class_names, examples_per_class, examples, class_names_to_ids


def write_data(path, datas):
    tmp = datas['support'] + datas['query']

    with open(path, 'w') as fout:
        for line in tmp:
            fout.write("%s\n" % json.dumps(line, ensure_ascii=False))  

class Dataloader():
    def __init__(self, vocabPath, filePath):
        super(Dataloader, self).__init__()

        self.class_names, self.examples_per_class, self.examples, self.class_names_to_ids = get_classes_infomation(vocabPath, filePath)
        
        self.dataset_spec = DatasetSpecification(
            name=None,
            images_per_class=self.examples_per_class,
            class_names=self.class_names,
            path=None,
            file_pattern='{}.tfrecords'
        )

        print(self.dataset_spec)


        self.config = EpisodeDescriptionConfig(
            num_ways = None,
            num_support = None,
            num_query = None,
            min_ways = 3,
            max_ways_upper_bound = 10,
            max_num_query = 20,
            max_support_set_size = 100,
            max_support_size_contrib_per_class = 20,
            min_log_weight = -0.69314718055994529,  # np.log(0.5),
            max_log_weight = 0.69314718055994529,  # np.log(2),
            min_examples_in_class = 2
            )
        self.sampler = EpisodeDescriptionSampler(
            dataset_spec=self.dataset_spec,
            split=Split.TRAIN,
            episode_descr_config=self.config,
            pool=None
            )

    def get_episode_datas(self):

        class_descriptions = self.sampler.sample_episode_description()
        episode_datas = defaultdict(list)
        
        class_ids, num_support, support_examples, query_examples = [], [], [], []
        num_classes = len(class_descriptions)
        num_query = class_descriptions[0][2]

        for class_des in class_descriptions:
            class_ids.append(class_des[0])
            num_support.append(class_des[1])

            examples_per_class = self.examples[self.class_names[class_des[0]]]
            np.random.seed(np.random.randint(0, 100000))
            np.random.shuffle(examples_per_class)
            support_per_class = examples_per_class[:class_des[1]]
            query_per_class = examples_per_class[class_des[1]: class_des[1]+class_des[2]]

            support_examples.extend(support_per_class)
            query_examples.extend(query_per_class)

        episode_datas['support'] = support_examples
        episode_datas['query'] = query_examples
        # print(f"Number of classes: {num_classes}")
        # print(f"List of class_ids: {str(class_ids)}")
        # print(f"List of support numbers: {str(num_support)}")
        # print(f"Number of query examples: {num_query}")
        # print(f"Support size: {len(support_examples)}")
        # print(f"Query size: {len(query_examples)}")
        # print(episode_datas)
        # write_data("code/slot_and_intent/data-process/test.json", episode_datas)
        
        return num_classes, class_ids, num_support, num_query, episode_datas

    def split_data(self, episode_datas):
        support_labels, support_text, query_labels, query_text, slots = [], [], [], [], []
        for line in episode_datas['support']:
            support_labels.append(self.class_names_to_ids[line['intent']])
            support_text.append(line['text_u'])
            slots.append(line["slots"].split(', '))
        for line in episode_datas['query']:
            
            query_labels.append(line["intent"])
            query_text.append(line["text_u"])
            slots.append(line["slots"].split(', '))
        return support_labels, support_text, query_labels, query_text, slots
        
        


