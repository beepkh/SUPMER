# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implements processors to convert examples to input and outputs, this can be
with integrarting patterns/verbalizers for PET or without."""
import abc 
import string 
from collections import OrderedDict

from .utils import Text, get_verbalization_ids, remove_final_punctuation, lowercase 

class AbstractProcessor(abc.ABC):
    def __init__(self, tokenizer, with_pattern, pattern_id=None, mask_position=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.mask_token_id
        self.mask_token = tokenizer.mask_token
        self.with_pattern = with_pattern 
        self.pattern_id = pattern_id
        self.tokenized_verbalizers = None  
        self.mask_position = mask_position

    def get_sentence_parts(self, example, mask_length):
        pass 

    def get_prompt_parts(self, example, mask_length):
        pass 

    def get_verbalizers(self):
        pass 

    def get_target(self, example):
        return example["label"]

    def get_tokenized_verbalizers(self, example=None):
        """If verbalizers are fixed per examples, this returns back a computed tokenized 
        verbalizers, but if this is example dependent, it computes the tokenized verbalizers
        per example. In this function, as a default, we compute the static one."""
        if self.tokenized_verbalizers is not None:
            return self.tokenized_verbalizers

        verbalizers = self.get_verbalizers()
        assert len(verbalizers) != 0, "for using static tokenized verbalizers computation, the length"
        "of verbalizers cannot be empty."
        self.tokenized_verbalizers=[[get_verbalization_ids(word=verbalizer, tokenizer=self.tokenizer)] for verbalizer in verbalizers]
        return self.tokenized_verbalizers

    def get_extra_fields(self, example=None):
        # If there is a need to keep extra information, here we keep a dictionary
        # from keys to their values.
        return {} 

    def get_classification_parts(self, example):
        pass 
   
    def get_parts_with_setting_masks(self, part_0, part_1, masks):
        "Only used in case of two sentences: 0`: [p,h,m],[]  `1`: [p,m,h],[]  `2`: [p],[m,h] , `3`: [p],[h,m]"
        if self.mask_position == '0':
            return part_0+part_1+masks, []
        elif self.mask_position == '1':
            return part_0+masks+part_1, []
        elif self.mask_position == '2':
            return part_0, masks+part_1
        elif self.mask_position == '3':
            return part_0, part_1+masks 
    
    

class MR(AbstractProcessor):
    name = "mr"
 
    def get_encoder_parts(self, example):
        source=Text(text=example["source"], shortenable=True)
        if self.pattern_id >= 0:
            text = "It was <extra_id_0> ?"
            return [source, Text(text=text)], []
        else:
            return [source], []

    def get_decoder_parts(self, idx):
        gt = self.get_verbalizers()[idx]
        return [Text(text="<extra_id_0>"), Text(text=gt)], [1,1]
    
    def get_verbalizers(self):
        return ["terrible", "great"]


class CR(MR):
    name="cr"


class SST2(MR):
    name = "SST-2"
 

class SST5(MR):
    name = "sst-5"
 
    def get_verbalizers(self):
        return ["terrible", "bad", "okay", "good", "great"]


class Subj(MR):
    name = "subj"
    
    def get_encoder_parts(self, example):
        source=Text(text=example["source"], shortenable=True)
        if self.pattern_id == 0:
            return [source, Text(text="This is"), Text(text="<extra_id_0> .")], []
        elif self.pattern_id == 1:
            return [source, Text(text="It's all"), Text(text="<extra_id_0> .")], []
        elif self.pattern_id == 2:
            return [source, Text(text="It is"), Text(text="<extra_id_0> .")], []
        elif self.pattern_id == 3:
            return [source, Text(text="Is it"), Text(text="<extra_id_0> ?")], []
        
    def get_verbalizers(self):
        return ["subjective", "objective"]  


class Trec(MR):
    name="trec"
    
    def get_encoder_parts(self, example):
        source = Text(text=example["source"], shortenable=True)
        if self.pattern_id == 0:
            return [source, Text(text="<extra_id_0> :")], []
        elif self.pattern_id == 1:
            return [source, Text(text="Q:"), Text(text="<extra_id_0> :")], []
        elif self.pattern_id == 2:
            return [source, Text(text="why"), Text(text="<extra_id_0> ?")] , []
        elif self.pattern_id == 3:
            return [source, Text(text="Answer:"), Text(text="<extra_id_0> .")], []
        
    def get_verbalizers(self):
        return  ["description", "entity", "expression", "human", "location", "number"]
    

class BoolQ(MR):
    name = "boolq"
    
    def get_encoder_parts(self, example):
        passage = Text(text=example["passage"], shortenable=True)
        question = Text(text=example["question"], shortenable=True)
        if self.pattern_id < 2:
            return [passage, Text(text='. Question: '), question, 
                   Text(text='? Answer: '), Text(text='<extra_id_0> .')], []
        elif self.pattern_id < 4:
            return [passage, Text(text='. Based on the previous passage, '), 
                    question, Text(text='?'), Text(text='<extra_id_0> .')], []
        else:
            return [Text(text='Based on the following passage, '), question,
                    Text(text='?'), Text(text='<extra_id_0> .'), passage], []
    
    def get_verbalizers(self):
        return ["yes", "no"]


class RTE(MR):
    name = "rte"
    
    def get_encoder_parts(self, example):
        premise = Text(text=example["premise"], shortenable=True)
        hypothesis = Text(text=example["hypothesis"].rstrip(string.punctuation), shortenable=True)
        hypothesis_with_punctuation = Text(text=example["hypothesis"], shortenable=True)
    
        if self.pattern_id == 0:
            return [Text(text='"'), hypothesis, Text(text='" ?')], [Text(text='<extra_id_0> , "'), premise, Text(text='"')]
        elif self.pattern_id == 1:
            return [hypothesis, Text(text='?')], [Text(text='<extra_id_0> ,'), premise]
        if self.pattern_id == 2:
            return [Text(text='"'), hypothesis, Text(text='" ?')], [Text(text='<extra_id_0> . "'), premise, Text(text='"')]
        elif self.pattern_id == 3:
            return [hypothesis, Text(text='?')], [Text(text='<extra_id_0> .'), premise]
        elif self.pattern_id == 4:
            return [premise, Text(text=' question: '),  hypothesis_with_punctuation,
                    Text(text=' True or False? answer: <extra_id_0>')], []
    
    def get_verbalizers(self):
        if self.pattern_id == 4:
            return ["true", "false"]
        else:
            return ["yes", "no"]


class CB(RTE):
    task = 'cb'
    
    def get_encoder_parts(self, example):
        if self.pattern_id == 4: 
            premise = Text(text=example["premise"], shortenable=True)
            hypothesis = Text(text=example["hypothesis"], shortenable=True)
            return [premise, Text(text=' question: '), hypothesis, Text(text=' true, false or neither? answer: <extra_id_0>')], []
        return super().get_encoder_parts(example) 
    
    def get_verbalizers(self):
        if self.pattern_id == 4:
            return ["true", "false", "neither"]
        else:
            return ["yes", "no", "maybe"]


class WiC(MR):
    name = "wic"
    
    def get_encoder_parts(self, example):
        sentence1 = Text(text=example["sentence1"], shortenable=True)
        sentence2 = Text(text=example["sentence2"], shortenable=True)
        word = example["word"]

        if self.pattern_id == 0:
            return [Text(text='"'), sentence1, Text(text='" / "'), sentence2,
                    Text(text='" Similar sense of "'+word+'"?'), Text(text='<extra_id_0> .')], []
        if self.pattern_id == 1:
            return [sentence1, sentence2, Text(text='Does ' + word + ' have the same meaning in both sentences? <extra_id_0>')], []
        if self.pattern_id == 2:
            return [Text(text=example["word"]), Text(text=' . Sense (1) (a) "'), sentence1,
                    Text(text='" ( <extra_id_0> ) "'), sentence2, Text(text='"')], []
        
    def get_verbalizers(self):
        if self.pattern_id == 2:
            return ["2", "b"]
        return ["No", "Yes"]


class QNLI(MR):
    name = "qnli"
    
    def get_encoder_parts(self, example):
        sentence = Text(text=example["sentence"], shortenable=True)
        question = Text(text=example["question"], shortenable=True)
        if self.pattern_id < 2:
            return [sentence, Text(text='. Question: '), question, 
                   Text(text='? Answer: '), Text(text='<extra_id_0> .')], []
        elif self.pattern_id < 4:
            return [sentence, Text(text='. Based on the previous sentence, '), 
                    question, Text(text='?'), Text(text='<extra_id_0> .')], []
        else:
            return [Text(text='Based on the following sentence, '), question,
                    Text(text='?'), Text(text='<extra_id_0> .'), sentence], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["Yes", "No"]
        return ["true", "false"]
    
    
class QQP(MR):
    name="qqp"
    
    def get_encoder_parts(self, example):
        question1 = Text(text=example["question1"], shortenable=True)
        question2 = Text(text=example["question2"], shortenable=True)
        if self.pattern_id < 2:
            return [Text(text='Do '), question1, Text(text=' and '), question2, 
                   Text(text=' have the same meaning? '), Text(text='<extra_id_0> .')], []
        elif self.pattern_id < 4:
            return [question1, Text(text='. Based on the previous question, '), 
                    question2, Text(text='?'), Text(text='<extra_id_0> .')], []
        else:
            return [Text(text='Based on the following question, '), question1,
                    Text(text='?'), Text(text='<extra_id_0> .'), question2], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["No", "Yes"]
        return ["false", "true"]


class MRPC(MR):
    name = "mrpc"
    
    def get_encoder_parts(self, example):
        sentence1 = Text(text=example["sentence1"], shortenable=True)
        sentence2 = Text(text=example["sentence2"], shortenable=True)
        if self.pattern_id < 2:
            return [Text(text='Do '), sentence1, Text(text=' and '), sentence2, 
                   Text(text=' have the same meaning? '), Text(text='<extra_id_0> .')], []
        elif self.pattern_id < 4:
            return [sentence1, Text(text='. Based on the previous sentence, '), 
                    sentence2, Text(text='?'), Text(text='<extra_id_0> .')], []
        else:
            return [Text(text='Based on the following sentence, '), sentence1,
                    Text(text='?'), Text(text='<extra_id_0> .'), sentence2], []

    def get_verbalizers(self):
        if self.pattern_id in [0, 2, 4]:
            return ["No", "Yes"]
        return ["false", "true"]
    

PROCESSOR_MAPPING = OrderedDict(
    [
        ('mr', MR),
        ('cr', CR),
        ('subj', Subj),
        ('trec', Trec),
        ('SST-2', SST2),
        ('sst-5', SST5),
        #superglue datasets 
        ('boolq', BoolQ),
        ('rte', RTE),
        ('cb', CB),
        ('wic', WiC),
        #glue datasets 
        ('qnli', QNLI),
        ('qqp', QQP),
        ('mrpc', MRPC)
    ]
)


class AutoProcessor:
    @classmethod
    def get(self, task, tokenizer, with_pattern, pattern_id, mask_position):
        if task in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[task](
                tokenizer=tokenizer,
                with_pattern=with_pattern,
                pattern_id=pattern_id,
                mask_position=mask_position)
        raise ValueError(
            "Unrecognized task {} for AutoProcessor: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in PROCESSOR_MAPPING.keys())
            )
        )