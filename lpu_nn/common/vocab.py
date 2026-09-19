#!/usr/bin/env python3

# system
import os
import random
import unicodedata
from collections import OrderedDict

# 3rd party
import sentencepiece as spm
import torch

# local
from lpu.common import logging
from lpu.common import progress
from lpu_nn.common.tokenizer import train_tokenizer

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

dict_str2vector = {}

def import_vectors(path):
    #dprint(path)
    logger.info(f"loading pre-trained vectors from: {path}")
    fobj = progress.view(path, "loading vectors")
    for line in fobj:
        #dprint(line)
        fields = line.strip().split(' ')
        if len(fields) >= 2:
            symbol = fields[0]
            symbol_lower = symbol.lower()
            symbol_norm = unicodedata.normalize('NFKC', symbol)
            symbol_lower_norm = unicodedata.normalize('NFKC', symbol_lower)
            vector = [float(f) for f in fields[1:]]
            vector = torch.tensor(vector)
            dict_str2vector[symbol] = vector
            #if symbol not in dict_str2vector:
            #    # prioritize first (more frequent)
            #    dict_str2vector[symbol] = vector
            if symbol_lower not in dict_str2vector:
                dict_str2vector[symbol_lower] = vector
            if symbol_norm not in dict_str2vector:
                dict_str2vector[symbol_norm] = vector
            if symbol_lower_norm not in dict_str2vector:
                dict_str2vector[symbol_lower_norm] = vector
        #break
    return dict_str2vector

class IDMapBase:
    def clean_ids(self, ids):
        ids = list(ids)
        if self.eos in ids:
            ids = ids[:ids.index(self.eos)]
        if ids[0:1] == [self.bos]:
            #ids = ids[1:]
            ids.pop(0)
        while ids[-1:] == [self.pad]:
            #ids = ids[0:-1]
            ids.pop(-1)
        return ids

    def get_state(self):
        return None

    def set_state(self, state):
        return None

class Vocabulary(IDMapBase):
    def __init__(self):
        self.sp = None
        #self.symbols = dict()

    def decode(self, elements, remove_symbols=True, as_tokens=False):
        try:
            elements = list(elements)
            if len(elements) == 0:
                return ''
            elif isinstance(elements[0], int):
                if remove_symbols:
                    return self.sp.decode_ids(elements)
                elif as_tokens:
                    #return str.join(' ', map(self.sp.id_to_piece, elements))
                    #dprint(elements)
                    #dprint([self.sp.id_to_piece(elem) for elem in elements])
                    return [self.sp.id_to_piece(elem) for elem in elements]
                else:
                    return str.join('', map(self.sp.id_to_piece, elements)).replace('▁', ' ').strip()
            elif isinstance(elements[0], str):
                if remove_symbols:
                    return self.sp.decode_pieces(elements)
                elif as_tokens:
                    #return str.join(' ',elements)
                    return elements
                else:
                    return str.join('',elements).replace('▁', ' ').strip()
            else:
                raise TypeError(f"unknown piece type: {type(elements[0])}")
        except Exception as e:
            dprint(elements)
            logger.warning(repr(elements))
            raise e

    #def encode(self, sent, to='ids', add_symbols=False, add_dummy_prefix=True):
    def encode(self, sent, to='ids', add_symbols=False, add_dummy_prefix=False):
        """
        :param str sent:
        :param str to:
        :param bool add_symbols:
        :rtype: list of int
        :return: list of ids or tokens
        """
        if add_dummy_prefix:
            if not sent[:1].isspace():
                sent = " " + sent
        if to == 'ids':
            ids = self.sp.encode_as_ids(sent)
        elif to in ['pieces', 'tokens']:
            return self.sp.encode_as_pieces(sent)
        else:
            raise ValueError(f"unknown encode target: {to}")
        if add_symbols:
            return self.safe_add_symbols(ids)
        else:
            return ids

    def convert(self, sent, to='ids'):
        if isinstance(sent, str):
            if to == 'str':
                # as-is
                return sent
            elif to == 'ids':
                return self.sp.encode_as_ids(sent)
            elif to in ['pieces', 'tokens']:
                return self.sp.encode_as_pieces(sent)
            else:
                raise ValueError(f"unknown encode target: {to}")
        elif isinstance(sent, (list,tuple)):
            t = type(sent)
            if to in (str, 'str'):
                return self.decode(sent)
            elif len(sent) == 0:
                return t()
            elif isinstance(sent[0], str):
                if to in ['pieces', 'tokens']:
                    return sent
                elif to == 'ids':
                    return t(self.sp.piece_to_id(piece) for piece in sent)
                else:
                    raise ValueError(f"unknown decode target: {to}")
            elif isinstance(sent[0], int):
                if to == 'ids':
                    return sent
                if to in ['pieces', 'tokens']:
                    return t(self.sp.id_to_piece(id) for id in sent)
                else:
                    raise ValueError(f"unknown decode target: {to}")
        else:
            raise ValueError(f"unsupported input type: {type(sent)}")

    def get_state(self):
        state = self.__dict__.copy()
        state['sp'] = None
        return state

    def load(self, path):
        logger.info(f"loading SentencePiece model: {path}")
        return self.loads(open(path, 'rb').read())

    def loads(self, buf):
        sp = spm.SentencePieceProcessor()
        sp.load_from_serialized_proto(buf)
        self.sp = sp
        self.sp_bytes = buf
        return self

    def remove_unk(self, sent):
        if isinstance(sent, str):
            return sent.replace('<unk>', '')
        elif isinstance(sent, (list,tuple)):
            if len(sent) == 0:
                return sent
            elif isinstance(sent[0], int):
                return type(sent)(id for id in sent if id != self.unk)
            elif isinstance(sent[0], str):
                return type(sent)(token for token in sent if token != '<unk>')
            else:
                raise TypeError(f"unsupported token type: {type(sent[0]).__class__.__name__}")
        else:
            raise TypeError(f"unsupported type: {type(sent).__class__.__name__}")

    def safe_add_symbols(self, ids, add_bos=True, add_eos=True):
        ids = list(ids)
        if add_bos:
            if ids[0:1] != [self.bos]:
                ids = [self.bos, *ids]
        if add_eos:
            if ids[-1:] != [self.eos]:
                ids = [*ids, self.eos]
        return ids

    def sample(self, exclude_symbols=True, additions=None):
        int_from = 0
        int_to = len(self.sp) - 1
        if exclude_symbols:
            int_from = len(self.symbols)
        if additions is not None:
            #if not isinstance(additions, (list, tuple)):
            if isinstance(additions, int):
                additions = [additions]
            if random.random() < len(additions) / float(len(additions) + (int_to - int_from + 1)):
                return random.choice(additions)
        return random.randint(int_from, int_to)

    def set_symbols(self, extra_symbols=None):
        default_symbols={"bos": '<s>', "eos": '</s>', "pad": '<pad>', "unk": '<unk>'}
        if hasattr(self, 'symbols'):
            symbols = self.symbols
        else:
            symbols = default_symbols
            self.bos = self.sp.bos_id()
            self.eos = self.sp.eos_id()
            self.pad = self.sp.pad_id()
            self.unk = self.sp.unk_id()
        if extra_symbols:
            symbols.update(extra_symbols)
            for key, sym in extra_symbols.items():
                id = self.sp.piece_to_id(sym)
                #if id == 0:
                if id == self.unk:
                    raise ValueError(f"unknown symbols: {sym}")
                setattr(self, key, id)
        self.symbols = symbols
        dprint(symbols)
        return self

    def set_state(self, state):
        self.__dict__.update(state)
        if 'sp_bytes' in state:
            self.loads(self.sp_bytes)
        return self

    def __len__(self):
        return len(self.sp)

    def __iter__(self):
        for i in range(len(self)):
            yield self.sp.id_to_piece(i)

class CharacterMap(IDMapBase):
    def __init__(self):
        self.set_symbols()

    def decode(self, elements, remove_symbols=True, as_tokens=False):
    #def decode(self, elements, remove_symbols=True):
    #def decode(self, elements):
        try:
            elements = list(elements)
            if len(elements) == 0:
                return ''
            elif isinstance(elements[0], int):
                if remove_symbols:
                    elements = self.sp.decode_pieces(elements)
                if as_tokens:
                    return bytes(elements)
                elements = self.clean_ids(elements)
                codes = [code - self.offset for code in elements]
                return str(bytes(codes), 'utf-8', errors='backslashreplace')
            #elif isinstance(elements[0], str):
            #    if remove_symbols:
            #        elements = self.clean_ids(elements)
            #    if as_tokens:
            #        return elements
            #    return str.join('',elements)
            else:
                raise TypeError(f"unknown piece type: {type(elements[0])}")
        except Exception as e:
            dprint(elements)
            logger.warning(repr(elements))
            raise e

    #def encode(self, sent, to='ids', add_symbols=False, add_dummy_prefix=True):
    def encode(self, sent, to='ids', add_symbols=False, add_dummy_prefix=False):
        """
        :param str sent:
        :param str to:
        :param bool add_symbols:
        :rtype: list of int
        :return: list of ids or tokens
        """
        if add_dummy_prefix:
            if not sent[:1].isspace():
                sent = " " + sent
        if to == 'ids':
            codes = bytes(sent, 'utf-8')
            ids = [code + self.offset for code in codes]
        elif to in ['tokens']:
            #return sent.split('')
            return bytes(sent, 'utf-8')
        else:
            raise ValueError(f"unknown encode target: {to}")
        if add_symbols:
            return self.safe_add_symbols(ids)
        else:
            return ids

    def convert(self, sent, to='ids'):
        if isinstance(sent, str):
            if to == 'str':
                # as-is
                return sent
            elif to == 'ids':
                return self.encode(sent, to='ids')
            elif to in ['tokens']:
                #return sent.split('')
                return bytes(sent, 'utf-8')
            else:
                raise ValueError(f"unknown encode target: {to}")
        elif isinstance(sent, (list,tuple)):
            t = type(sent)
            if to in (str, 'str'):
                return self.decode(sent)
            elif len(sent) == 0:
                return t()
            #elif isinstance(sent[0], str):
            #    if to in ['tokens']:
            #        return sent
            #    elif to == 'ids':
            #        return t(self.sp.piece_to_id(piece) for piece in sent)
            #    else:
            #        raise ValueError("unknown decode target: {}".format(to))
            elif isinstance(sent[0], int):
                if to == 'ids':
                    return sent
                if to in ['tokens']:
                    codes = [code - self.offset for code in sent]
                    return bytes(codes)
                else:
                    raise ValueError(f"unknown decode target: {to}")
        else:
            raise ValueError(f"unsupported input type: {type(sent)}")

    def safe_add_symbols(self, ids, add_bos=True, add_eos=True):
        ids = list(ids)
        if add_bos:
            if ids[0:1] != [self.bos]:
                ids = [self.bos, *ids]
        if add_eos:
            if ids[-1:] != [self.eos]:
                ids = [*ids, self.eos]
        return ids

    def sample(self, exclude_symbols=True, additions=None):
        int_from = 0
        int_to = len(self.sp) - 1
        if exclude_symbols:
            int_from = len(self.symbols)
        if additions is not None:
            #if not isinstance(additions, (list, tuple)):
            if isinstance(additions, int):
                additions = [additions]
            if random.random() < len(additions) / float(len(additions) + (int_to - int_from + 1)):
                return random.choice(additions)
        return random.randint(int_from, int_to)

    def set_symbols(self, extra_symbols=None):
        #default_symbols=OrderedDict([('bos', '<s>'), ('eos', '</s>')])
        default_symbols=['bos', 'eos']
        if hasattr(self, 'symbols'):
            symbols = self.symbols
        else:
            symbols = default_symbols
        if extra_symbols:
            for symbol in extra_symbols:
                if symbol not in symbols:
                    symbols.append(symbol)
        for i, symbol in enumerate(symbols):
            setattr(self, symbol, i)
        self.symbols = symbols
        self.offset = len(symbols)
        dprint(symbols)
        return self

    def __len__(self):
        return 256 + self.offset

    def __iter__(self):
        yield from self.list_id2str

class IDMap(IDMapBase):
    def __init__(self, sep=' '):
        self.list_id2str = []
        self.dict_str2id = {}
        self.dict_count = {}
        self.set_symbols()
        self.sep = sep

    def id2str(self, i):
        if i in range(len(self)):
            return self.list_id2str[i]
        else:
            return '<unk>'

    def str2id(self, s, growth=False):
        #dprint(s)
        if growth:
            self.dict_count[s] = self.dict_count.get(s, 0) + 1
        if s in self.dict_str2id:
            return self.dict_str2id[s]
        elif growth:
            new_id = len(self.list_id2str)
            self.list_id2str.append(s)
            self.dict_str2id[s] = new_id
            return new_id
        else:
            return self.unk

    def set_symbols(self, extra_symbols=None):
        symbol_pairs = [('pad', '<pad>'), ('bos','<s>'), ('eos','</s>'), ('unk', '<unk>')]
        symbols = OrderedDict(symbol_pairs)
        if extra_symbols:
            symbols.update(extra_symbols)
        for key, sym in symbols.items():
            id = self.str2id(sym, True)
            setattr(self, key, id)
        self.symbols = symbols
        #dprint(symbols)
        return self

    def safe_add_symbols(self, ids, add_bos=True, add_eos=True):
        ids = list(ids)
        if add_bos:
            if ids[0:1] != [self.bos]:
                ids = [self.bos, *ids]
        if add_eos:
            if ids[-1:] != [self.eos]:
                ids = [*ids, self.eos]
        return ids

    #def encode(self, string, add_symbols=False):
    def encode(self, string, add_symbols=False):
        """
        :param str sent:
        :param str to:
        :param bool add_symbols:
        :rtype: list of int
        :return: list of ids or tokens
        """
        if self.sep is None:
            tokens = [string]
        else:
            tokens = string.split(self.sep)
        #ids = [self.str2id[token] for token in tokens]
        ids = [self.str2id(token) for token in tokens]
        if add_symbols:
            ids = self.safe_add_symbols(ids)
        if self.sep is None:
            return tokens[0]

    def decode(self, ids, remove_symbols=True, as_tokens=False):
        if isinstance(ids, int):
            ids = [ids]
        tokens = [self.id2str(id) for id in ids]
        if as_tokens:
            return tokens
        elif self.sep:
            return str.join(self.sep, tokens)
        else:
            return tokens[0]
        #if as_tokens or self.sep is None:
        #    return tokens[0]
        #else:
        #    return str.join(self.sep, tokens)

    def feed_field(self, string):
        if self.sep is None:
            self.str2id(string, growth=True)
        for token in string.split(self.sep):
            self.str2id(token, growth=True)

    def feed_corpus(self, tsv_path, indices, field_sep='\t'):
        if isinstance(indices, int):
            indices = [indices]
        for line in open(tsv_path):
            #dprint(line)
            #dprint(repr(field_sep))
            fields = line.strip().split(field_sep)
            #dprint(fields)
            for i in indices:
                #dprint(i)
                #dprint(fields[i])
                #self.str2id(fields[i], growth=True)
                self.feed_field(fields[i])
        return True

    def truncate(self, vocab_size):
        self.list_id2str = []
        self.dict_str2id = {}
        set_symbols = set(self.symbols.values())
        for sym in self.symbols.values():
            self.str2id(sym, growth=True)
        #for key, val in sorted(self.dict_count.items(), key=lambda k: -k[1]):
        for token, _count in sorted(self.dict_count.items(), key=lambda k: -k[1]):
            if token in set_symbols:
                continue
            if len(self) >= vocab_size:
                break
            self.str2id(token, growth=True)
        return self

    def set_state(self, state):
        self.__dict__.update(state)
        return self

    def get_state(self):
        state = self.__dict__.copy()
        return state

    def save(self, path):
        with open(path, 'w') as fobj:
            set_symbols = set(self.symbols.values())
            #dprint(set_symbols)
            for _i, token in enumerate(self.list_id2str):
                #fobj.write("{}\t{}\n".format(i, token))
                if token in set_symbols:
                    count = 0
                else:
                    count = self.dict_count.get(token, 0)
                fobj.write(f"{count}\t{token}\n")

    def load(self, path):
        for line in open(path):
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 2:
                count, token = fields
                #assert id == self.str2id(token, growth=True)
                self.str2id(token, growth=True)
                self.dict_count[token] = int(count)
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.str2id(key)
        elif isinstance(key, int):
            return self.id2str(key)
        else:
            raise KeyError(f"unsupported type: {type(key).__name__}")

    def __len__(self):
        return len(self.list_id2str)

    def __iter__(self):
        yield from self.list_id2str

class LabelMap(IDMap):
    def __init__(self):
        super().__init__(sep=None)

    def feed_field(self, string):
        #dprint(string)
        for label in string.split(self.sep):
            #dprint("--")
            #dprint(label)
            components = label.split(':')
            if len(components) == 2:
                try:
                    float(components[1])
                    label = components[0]
                except Exception as e:
                    dprint(e)
            #dprint(label)
            self.str2id(label, growth=True)

    def str2dist(self, string):
        #dprint(string)
        dist = [0.0] * len(self.list_id2str)
        #num_assign = 0
        #list_labels = string.split(self.sep)
        #for label in list_labels:
        total = 0
        for label in string.split(self.sep):
            #assigned = False
            components = label.split(':')
            if len(components) == 2:
                try:
                    prob = float(components[1])
                    label = components[0]
                    id = self.str2id(label, growth=True)
                    if prob >= 0.0:
                        dist[id] += prob
                        total += prob
                except Exception as e:
                    dprint(e)
            if total == 0:
                id = self.str2id(label, growth=True)
                dist[id] += 1.0
                total += 1.0
        if hasattr(self, 'unk'):
            if total < 1.0:
                # fill remain for <unk>
                dist[self.unk] += (1.0 - total)
        elif total > 1.0:
            # normalizing
            for i, prob in enumerate(dist):
                dist[i] = prob / total
        #dprint(dist)
        return dist

    def str2score(self, string):
        if isinstance(string, int):
            return string
        dist = self.str2dist(string)
        score = 0
        for label, prob in zip(self.list_id2str, dist, strict=False):
            try:
                s = float(label)
            except ValueError:
                # 数値でないラベルは期待値に寄与しない
                continue
            score += s * prob
        return score

    def set_symbols(self, extra_symbols=None, add_unk=False):
        #symbol_pairs = [('unk', '<unk>')]
        symbol_pairs = []
        if add_unk:
            symbol_pairs.append( ('unk', '<unk>') )
        symbols = OrderedDict(symbol_pairs)
        if extra_symbols:
            symbols.update(extra_symbols)
        for key, sym in symbols.items():
            id = self.str2id(sym, True)
            setattr(self, key, id)
        self.symbols = symbols
        return self

class FieldMap:
    def __init__(self):
        #self.map_dict = {}
        self.dict_maps = {}
        self.main_fields = OrderedDict()

    @classmethod
    def train(cls, workdir, main_fields, tsv_path, vocab_size, extra_symbols):
        if not os.path.isfile(os.path.join(workdir, 'sp.model')):
            model_prefix = os.path.join(workdir, 'sp')
            seq_indices = []
            for i, (_key, val) in enumerate(main_fields.items()):
                if val in ['seq']:
                    seq_indices.append(i)
            if seq_indices:
                #Vocabulary.train_tsv(model_prefix, tsv_path, vocab_size, extra_symbols, seq_indices)
                #train_tokenizer(model_prefix, tsv_path, vocab_size, extra_symbols)
                train_tokenizer(model_prefix, tsv_path, vocab_size)
        for i, (key, val) in enumerate(main_fields.items()):
            #if val in ['tokens']:
            dprint( (key, val) )
            if val in ['tags']:
                logger.debug(f"feeding corpus for field map: {key} ({val})")
                save_path = f"{workdir}/map_{key}.txt"
                idmap = IDMap()
                idmap.set_symbols()
                #idmap.set_symbols(extra_symbols)
                #dprint([i])
                idmap.feed_corpus(tsv_path, [i])
                #idmap.truncate(vocab_size)
                idmap.save(save_path)
            if val in ['label']:
                logger.debug(f"feeding corpus for field map: {key} ({val})")
                save_path = f"{workdir}/map_{key}.txt"
                idmap = LabelMap()
                idmap.set_symbols()
                idmap.feed_corpus(tsv_path, [i])
                idmap.save(save_path)
        return True

    def encode_phrase_tag(self, seq_field, substr, tag_field, tag, to='ids'):
        vocab = self.dict_maps[seq_field]
        idmap = self.dict_maps[tag_field]
        def to_tag_token(tag):
            if to == 'ids':
                return idmap.str2id(tag)
            else:
                return tag # as-is
        #dprint(substr)
        #dprint(tag)
        assert isinstance(vocab, Vocabulary)
        assert isinstance(idmap, IDMap)
        #id_seq = vocab.encode(substr, 'ids', add_symbols=False)
        #token_seq = vocab.encode(substr, to, add_symbols=False)
        token_seq = vocab.encode(substr, to, add_symbols=False, add_dummy_prefix=False)
        #tag_id = idmap.str2id(tag)
        tag_token = to_tag_token(tag)
        #tag_ids = [tag_token] * len(token_seq)
        tag_tokens = [tag_token] * len(token_seq)
        if len(tag_tokens) >= 2:
            if tag.startswith('B-'):
                # begining-tag
                #dprint(substr)
                #dprint(tag)
                inside_tag = 'I-' + tag[2:]
                #inside_id = idmap.str2id(inside_tag)
                inside_token = to_tag_token(inside_tag)
                tag_tokens = [tag_token] + [inside_token] * (len(token_seq)-1)
                #dprint(id_seq)
                #dprint(tag_ids)
        #dprint(id_seq)
        #dprint(tag_ids)
        return token_seq, tag_tokens

    def encode_pair(self, seq_field, string, tag_field, tags, to='ids', sep=' '):
    #def encode_pair(self, seq_field, string, tag_field, tags, sep=' '):
        assert isinstance(string, str)
        assert isinstance(tags, str)
        #dprint(string)
        #dprint(tags)
        list_sub = string.split(' ')
        #list_sub = string.split(sep)
        list_tags = tags.split(' ')
        #dprint(list_sub)
        #dprint(list_tags)
        #dprint(len(list_sub))
        #dprint(len(list_tags))
        assert len(list_sub) == len(list_tags)
        #all_token_ids = []
        all_seq_tokens = []
        #all_tag_ids = []
        all_tag_tokens = []
        for sub, tag in zip(list_sub, list_tags, strict=False):
            if sep:
                #sub = " " + sub
                if not sub[:1].isspace():
                    sub = sep + sub
            #token_ids, tag_ids = self.encode_phrase_tag(seq_field, sub, tag_field, tag, to)
            seq_tokens, tag_tokens = self.encode_phrase_tag(seq_field, sub, tag_field, tag, to)
            all_seq_tokens  += seq_tokens
            all_tag_tokens += tag_tokens
            #if sub in ['SOCCER', 'FRENCH']:
            #    dprint("-----")
            #    dprint(sub)
            #    dprint(tag)
            #    dprint(token_ids)
            #    dprint(tag_ids)
        #dprint(self.dict_maps['t'].dict_str2id)
        return all_seq_tokens , all_tag_tokens

    def set_symbols(self, extra_symbols):
        for key, idmap in self.dict_maps.items():
            dprint(key)
            dprint(idmap)
            idmap.set_symbols(extra_symbols)
        return self

    def load(self, workdir, main_fields):
        for _i, (key, val) in enumerate(main_fields.items()):
            if val in ['seq']:
                if 'seq' not in self.dict_maps:
                    model_path = os.path.join(workdir, 'sp.model')
                    vocab = Vocabulary().load(model_path)
                    self.dict_maps['seq'] = vocab
                idmap = self.dict_maps['seq']
                self.dict_maps[key] = vocab
            #if val in ['tokens']:
            if val in ['tags']:
                list_path = f"{workdir}/map_{key}.txt"
                idmap = IDMap().load(list_path)
                self.dict_maps[key] = idmap
            if val in ['label']:
                list_path = f"{workdir}/map_{key}.txt"
                idmap = LabelMap().load(list_path)
                self.dict_maps[key] = idmap
        self.main_fields.update(main_fields)
        return self

    def get_state(self):
        state = {}
        state['main_fields'] = self.main_fields
        state['dict_maps'] = dict_maps = {}
        for key, val in self.dict_maps.items():
            dict_maps[key] = val.get_state()
        return state

    def set_state(self, state):
        self.main_fields = state['main_fields']
        for key, val in state['dict_maps'].items():
            #format = self.main_fields[key]
            format = self.main_fields.get(key)
            if key == 'seq':
                self.dict_maps[key] = Vocabulary().set_state(val)
            elif format == 'seq':
                self.dict_maps[key] = Vocabulary().set_state(val)
            elif format == 'tags':
                self.dict_maps[key] = IDMap().set_state(val)
            elif format == 'label':
                self.dict_maps[key] = LabelMap().set_state(val)
            else:
                raise KeyError(f"unknown variable name: {key}")
        return self

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __contains__(self, key):
        return key in self.dict_maps

    def __getitem__(self, key):
        return self.dict_maps[key]
