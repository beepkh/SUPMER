# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT2 style dataset."""

import time

import numpy as np
from .IndexDataset import make_dataset as make_indexed_dataset

def get_train_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 2:
        splits.append(0.)
    splits = splits[:2]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 3
    assert splits_index[-1] == size
    return splits_index


def build_train_test_datasets(tokenizer, data_class, data_prefix, data_impl, splits_string,
                              max_seq_length, max_dec_length, skip_warmup, domain, dataset_name):
    """Build train, valid, and test datasets."""
    # Indexed dataset.
    context_data_prefix = data_prefix + "_{}".format(domain) + "_context"
    context_indexed_dataset = get_indexed_dataset_(context_data_prefix,
                                                   data_impl,
                                                   skip_warmup)
    
    if dataset_name != 'ss':
        target_data_prefix = data_prefix + "_{}".format(domain) + "_target" 
        target_indexed_dataset = get_indexed_dataset_(target_data_prefix,
                                                      data_impl,
                                                      skip_warmup)
    else:
        target_data_prefix, target_indexed_dataset = None, None

    total_num_of_documents = context_indexed_dataset.sizes.shape[0]
    splits = get_train_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print(' > dataset split:')

    def print_split_stats(name, index):
        print('    {}:'.format(name))
        print('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('test', 1)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            document_ids_in_splits = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = data_class(tokenizer, name, domain, 
                                 document_ids_in_splits, context_indexed_dataset, target_indexed_dataset,
                                 max_seq_length, max_dec_length)
        return dataset

    train_dataset = build_dataset(0, 'train')
    test_dataset = build_dataset(1, 'test')

    return (train_dataset, test_dataset)

def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    print(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset