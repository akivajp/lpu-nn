#!/usr/bin/env python3

# system
import argparse
import datetime
import json
import os
import time
from zoneinfo import ZoneInfo

# 3rd
# bottle は任意依存 (serve エクストラ)。未導入でもモジュールの読み込みは
# 通るようにし、実際に起動するときに導線を示して終了する
try:
    from bottle import TEMPLATE_PATH, request, response, route, run
    from bottle import jinja2_template as template
    BOTTLE_IMPORT_ERROR: "ImportError | None" = None
except ImportError as error:  # pragma: no cover - 任意依存が無い環境
    BOTTLE_IMPORT_ERROR = error
    TEMPLATE_PATH = []
    request = response = None

    def route(*_args, **_kwargs):
        def decorate(func):
            return func
        return decorate

    def run(*_args, **_kwargs):
        raise RuntimeError("bottle is not installed")

    def template(*_args, **_kwargs):
        raise RuntimeError("bottle is not installed")

# local
from lpu.common import logging
from lpu_nn.commands import train_seq2seq

logger = logging.getColorLogger(__name__)
dprint = logger.debug_print

DEFAULT_PORT = 8000
DEFAULT_HOST = '127.0.0.1'
MAX_LENGTH = 500

#dirpath = os.path.dirname(os.path.abspath(__file__))
dirpath = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(dirpath)
# テンプレートはパッケージに同梱している
TEMPLATE_DIR = os.path.join(os.path.dirname(dirpath), 'server_static')
TEMPLATE_PATH.append(TEMPLATE_DIR)
dprint(TEMPLATE_PATH)

TIMEZONE = ZoneInfo('Asia/Tokyo')

def timestamp():
    """The current time in the configured zone

    設定した時間帯での現在時刻。

    従来は pytz で素の現在時刻 (実行環境の地方時) に日本時間の札を
    貼っていたため、日本時間以外の環境では誤った時刻を報告していた。
    標準ライブラリの zoneinfo で、時間帯を指定して現在時刻を取る。
    """
    return datetime.datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

##GPU_ID = 0
##GPU_ID = 1
#GPU_ID = -1
##BEAM_WIDTH = 1
##BEAM_WIDTH = 2
#BEAM_WIDTH = 3
##BEAM_WIDTH = 5
##BEAM_WIDTH = 10
#TRG_MAX_LENGTH = 100
#INCOMPLETE_COST = 100
#REPETITION_COST = 0
##REPETITION_COST = 1
##REPETITION_COST = 2
##REPETITION_COST = 3
##REPETITION_COST = 5
##REPETITION_COST = 10
##REPETITION_COST = 100
#MAX_STEPS=2
##NORMALIZE = False
#NORMALIZE = True
#LOGGING = "/home/akiva/translated.log"

@route('/')
def index():
    model_names = models.keys()
    context = {'model_names': model_names}
    return template('seq2seq.html', context)

@route('/api/decode', method=['GET', 'POST'])
def decode():
    params = request.params.decode()
    model_name = params.getunicode('model_name')
    input_list = params.getall('input')
    timeout = params.get('timeout', 10)
    dprint(timeout)
    timeout = float(timeout)
    dprint(input_list)
    dprint(type(input_list))
    result = {}
    result["input"] = input_list
    result["timeout"] = timeout
    result["model_name"] = model_name
    try:
        if not model_name:
            response.status = 500
            result["error"] = {'message': "Mandatory parameter 'model_name' is not given."}
        #elif not input:
        elif not input_list:
            response.status = 500
            result["error"] = {'message': "Mandatory parameter 'input' is not given."}
        elif model_name not in models:
            response.status = 500
            result["error"] = {'message': f"Unknown model name: {model_name}"}
        else:
            model = models[model_name]
            time_start = time.time()
            len_input = [len(input) for input in input_list]
            max_len = max(len_input)
            if max_len > MAX_LENGTH:
                response.status = 500
                result["error"] = {
                    'message': f"Too long input (max_length: {MAX_LENGTH})"}
                return result
            BEAM = False
            if BEAM: # beam
                pred = model.beam_search(
                    input_list,
                    beam_width=3,
                    max_length=max_len * 3,
                    incomplete_cost=0,
                    repetition_cost=0,
                    normalize=True,
                    timeout=timeout,
                )
                dprint(pred)
                trg_ids, score = pred[0]
                result['score'] = score
                output_list = model.restore_batch(trg_ids, str, squeeze=False)
            else:
                pred = model.generate(input_list, max_length=max_len*3, timeout=timeout)
                output_list = model.restore_batch(pred, str, squeeze=False)
            data = []
            for i, input in enumerate(input_list):
                rec = {}
                rec['len_input'] = len_input[i]
                rec['input'] = input
                output = output_list[i]
                rec['len_output'] = len(output)
                rec['output'] = output
                data.append(rec)
            result['data'] = data
            result["required_time"] = time.time() - time_start
            result["timestamp"] = timestamp()
            result["output"] = output_list
    except Exception as e:
        response.status = 500
        result["error"] = {'message': repr(e)}
    response.headers['Content-Type'] = 'application/json'
    response.headers['Cache-Control'] = 'no-cache'
    dprint(result)
    result_json = json.dumps(result)
    if logpath:
        with open(logpath, 'a', encoding='utf-8') as fobj:
            #fobj.write(result_json + "\n")
            fobj.write(repr(result) + "\n")
    return result_json

@route('/api/models', method=['GET', 'POST'])
def get_model_names():
    result = {}
    result["available_models"] = list( models.keys() )
    result["timestamp"] = timestamp()
    response.headers['Content-Type'] = 'application/json'
    response.headers['Cache-Control'] = 'no-cache'
    dprint(result)
    result_json = json.dumps(result)
    return result_json


def main():
    #global trainer
    #global model
    global trainers
    global models
    global logpath
    global MAX_LENGTH
    parser = argparse.ArgumentParser('Sequence-to-Sequence Server')
    #parser.add_argument('model_mapping', help='model mapping string ("name1=model_path1,name2=model_path2,...")')
    parser.add_argument('model_mapping', help='pair of model name and path with format as "model_name=model_path" (e.g. "trans_ja-en=~/models/trans_ja-en/record.best_dev_acc")', nargs='*')
    parser.add_argument('--port', '-P', type=int, default=DEFAULT_PORT)
    # 既定では localhost のみで待ち受ける。従来は 0.0.0.0 に固定されており、
    # 研究用の手元サーバがネットワークへそのまま露出していた
    parser.add_argument('--host', '-H', type=str, default=DEFAULT_HOST,
                        help=f'address to listen on (default: {DEFAULT_HOST})')
    parser.add_argument('--reloader', action='store_true',
                        help='restart the server when a source file changes')
    parser.add_argument('--logpath', '--log', '-L', type=str, default=None)
    parser.add_argument('--max-length', '-M', type=int, default=MAX_LENGTH)
    parser.add_argument('--gpu', '-G', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    if BOTTLE_IMPORT_ERROR is not None:
        logger.error(
            "this command needs bottle, which is an optional dependency: "
            "install it with `pip install 'lpu-nn[serve]'`")
        raise SystemExit(1)
    dprint(args)
    #trainer = train_seq2seq.Seq2SeqTrainer().load_status(args.model)
    trainers = {}
    models = {}
    #for keyval in args.model_mapping.split(','):
    for keyval in args.model_mapping:
        name, model = keyval.split('=')
        dprint(name)
        dprint(model)
        if name in models:
            raise RuntimeError(f"Model name is already registered: {name}")
        logger.info(f"loading model: {model}")
        trainer = train_seq2seq.Seq2SeqTrainer().load_status(model)
        trainers[name] = trainer
        models[name] = trainer.model
        if args.gpu >= 0:
            trainer.model.to(args.gpu)
        #dprint(trainer.config)
        dprint(trainer.config.to_json(indent=2))
        logger.info(f"loaded model: {model}")
    #model = trainer.model
    logpath = args.logpath
    MAX_LENGTH = args.max_length
    dprint(logpath)
    if not models:
        logger.error("please give at least one 'name=path' pair")
        raise SystemExit(1)
    logger.info(f"listening on http://{args.host}:{args.port}")
    # debug=True は例外の内容を要求元へ返すため、既定では有効にしない
    run(host=args.host, port=args.port, reloader=args.reloader)

if __name__ == '__main__':
    main()
