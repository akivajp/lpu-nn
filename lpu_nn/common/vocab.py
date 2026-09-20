#!/usr/bin/env python3

# system
import os
import random
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

# 3rd party
import sentencepiece as spm
import torch

# local
from lpu.common import logging
from lpu.common import progress
# IDMap and LabelMap used to be carried here as a copy of an older lpu, with
# defects of their own; lpu now holds the tested implementation.
# (IDMap と LabelMap は古い lpu の複製を持ち回っており、独自の不具合も
#  抱えていた。検証済みの実装は lpu 側にある)
from lpu.common.vocab import IDMap, LabelMap
from lpu_nn.common.tokenizer import train_tokenizer

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

# 事前学習済みの単語ベクトル。表層 (原形・小文字化・NFKC 正規化) -> ベクトル
dict_str2vector: dict[str, torch.Tensor] = {}

def import_vectors(path: str) -> dict[str, torch.Tensor]:
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
            vector = torch.tensor([float(f) for f in fields[1:]])
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

class Vocabulary:
    """A SentencePiece model wrapped with the special symbols of this package

    SentencePiece モデルを、本パッケージの特殊記号とともに包んだ語彙。
    """

    def __init__(self) -> None:
        # SentencePieceProcessor は load()/loads() まで未設定
        self.sp: Any = None
        #self.symbols = dict()
        # 記号名 -> 表層。空であることが「まだ set_symbols を通っていない」
        # ことを表す (以前は hasattr による判定だった)
        self.symbols: dict[str, str] = {}
        # 記号の ID。SentencePiece と同じく、未定義は -1 で表す
        self.bos: int = -1
        self.eos: int = -1
        self.pad: int = -1
        self.unk: int = -1

    def clean_ids(self, ids: Iterable[int]) -> list[int]:
        """Strip the BOS / EOS / PAD symbols from an ID sequence

        ID 列から BOS / EOS / PAD 記号を取り除く。
        最初の EOS 以降を捨て、先頭の BOS を取り除き、末尾に連続する
        PAD を切り詰める。
        """
        ids = list(ids)
        if self.eos in ids:
            ids = ids[:ids.index(self.eos)]
        if ids[0:1] == [self.bos]:
            ids.pop(0)
        while ids[-1:] == [self.pad]:
            ids.pop(-1)
        return ids

    def decode(self, elements: Iterable[Any], remove_symbols: bool = True,
               as_tokens: bool = False) -> Any:
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
    def encode(self, sent: str, to: str = 'ids', add_symbols: bool = False,
               add_dummy_prefix: bool = False) -> Any:
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

    def convert(self, sent: Any, to: Any = 'ids') -> Any:
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

    def get_state(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['sp'] = None
        return state

    def load(self, path: str) -> "Vocabulary":
        logger.info(f"loading SentencePiece model: {path}")
        return self.loads(open(path, 'rb').read())

    def loads(self, buf: bytes) -> "Vocabulary":
        sp = spm.SentencePieceProcessor()
        sp.load_from_serialized_proto(buf)
        self.sp = sp
        self.sp_bytes = buf
        return self

    def remove_unk(self, sent: Any) -> Any:
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

    def safe_add_symbols(self, ids: Iterable[int], add_bos: bool = True,
                         add_eos: bool = True) -> list[int]:
        ids = list(ids)
        if add_bos:
            if ids[0:1] != [self.bos]:
                ids = [self.bos, *ids]
        if add_eos:
            if ids[-1:] != [self.eos]:
                ids = [*ids, self.eos]
        return ids

    def sample(self, exclude_symbols: bool = True,
               additions: "int | Sequence[int] | None" = None) -> int:
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

    def set_symbols(self, extra_symbols: "Mapping[str, str] | None" = None) -> "Vocabulary":
        default_symbols={"bos": '<s>', "eos": '</s>', "pad": '<pad>', "unk": '<unk>'}
        # 既に記号が入っていれば、それを保ったまま追加分だけを反映する
        if self.symbols:
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

    def set_state(self, state: Mapping[str, Any]) -> "Vocabulary":
        self.__dict__.update(state)
        if 'sp_bytes' in state:
            self.loads(self.sp_bytes)
        return self

    def __len__(self) -> int:
        return len(self.sp)

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self.sp.id_to_piece(i)

class FieldMap:
    def __init__(self) -> None:
        #self.map_dict = {}
        # フィールド名 -> Vocabulary / IDMap / LabelMap
        self.dict_maps: dict[str, Any] = {}
        self.main_fields: dict[str, str] = OrderedDict()

    @classmethod
    def train(cls, workdir: str, main_fields: Mapping[str, str], tsv_path: str,
              vocab_size: int, extra_symbols: Mapping[str, str]) -> bool:
        if not os.path.isfile(os.path.join(workdir, 'sp.model')):
            model_prefix = os.path.join(workdir, 'sp')
            seq_indices = []
            for i, (_key, val) in enumerate(main_fields.items()):
                if val in ['seq']:
                    seq_indices.append(i)
            if seq_indices:
                #Vocabulary.train_tsv(model_prefix, tsv_path, vocab_size, extra_symbols, seq_indices)
                # extra_symbols を渡さないと、set_symbols() が
                # piece_to_id() で unk を引いて ValueError になる
                train_tokenizer(model_prefix, tsv_path, vocab_size,
                                user_defined_symbols=list(extra_symbols.values()))
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

    def encode_phrase_tag(self, seq_field: str, substr: str, tag_field: str,
                          tag: str, to: str = 'ids') -> tuple[list[Any], list[Any]]:
        vocab = self.dict_maps[seq_field]
        idmap = self.dict_maps[tag_field]
        def to_tag_token(tag: str) -> Any:
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

    def encode_pair(self, seq_field: str, string: str, tag_field: str, tags: str,
                    to: str = 'ids', sep: str = ' ') -> tuple[list[Any], list[Any]]:
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

    def set_symbols(self, extra_symbols: Mapping[str, str]) -> "FieldMap":
        for key, idmap in self.dict_maps.items():
            dprint(key)
            dprint(idmap)
            idmap.set_symbols(extra_symbols)
        return self

    def load(self, workdir: str, main_fields: Mapping[str, str]) -> "FieldMap":
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

    @staticmethod
    def _map_state(field_map: Any) -> dict[str, Any]:
        """Capture the state of one field map

        1 つのフィールドマップの状態を取り出す。

        `Vocabulary` knows how to leave its SentencePiece processor out;
        `IDMap` and `LabelMap` come from lpu and hold plain attributes, so
        their instance dictionary is the state. Reaching for the dictionary
        keeps the state handling here instead of widening lpu's API.

        `Vocabulary` は SentencePiece のプロセッサを除く方法を自分で知って
        いる。`IDMap` と `LabelMap` は lpu 由来で通常の属性のみを持つため、
        インスタンス辞書がそのまま状態になる。辞書を直接扱うことで、
        状態の扱いを lpu 側の API を広げずにここへ閉じ込める。
        """
        if hasattr(field_map, 'get_state'):
            return dict(field_map.get_state())
        return dict(vars(field_map))

    @staticmethod
    def _restore_map(field_map: Any, state: Mapping[str, Any]) -> Any:
        """Restore one field map from the state captured above

        上で取り出した状態から 1 つのフィールドマップを復元する。
        """
        if hasattr(field_map, 'set_state'):
            return field_map.set_state(state)
        vars(field_map).update(state)
        return field_map

    def get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        state['main_fields'] = self.main_fields
        state['dict_maps'] = dict_maps = {}
        for key, val in self.dict_maps.items():
            dict_maps[key] = self._map_state(val)
        return state

    def set_state(self, state: Mapping[str, Any]) -> "FieldMap":
        self.main_fields = state['main_fields']
        for key, val in state['dict_maps'].items():
            #format = self.main_fields[key]
            format = self.main_fields.get(key)
            if key == 'seq':
                self.dict_maps[key] = Vocabulary().set_state(val)
            elif format == 'seq':
                self.dict_maps[key] = Vocabulary().set_state(val)
            elif format == 'tags':
                self.dict_maps[key] = self._restore_map(IDMap(), val)
            elif format == 'label':
                self.dict_maps[key] = self._restore_map(LabelMap(), val)
            else:
                raise KeyError(f"unknown variable name: {key}")
        return self

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    def __contains__(self, key: object) -> bool:
        return key in self.dict_maps

    def __getitem__(self, key: str) -> Any:
        return self.dict_maps[key]
