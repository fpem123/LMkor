'''
    Name: main.py
    Writer: Hoseop Lee, Ainizer
    Rule: Flask app server
    update: 21.03.03
'''

from transformers import TFGPT2LMHeadModel, BertTokenizerFast, BertForMaskedLM, EncoderDecoderModel, AlbertForMaskedLM
from flask import Flask, request, jsonify, render_template
import torch

from queue import Queue, Empty
from threading import Thread
import string
import time

from examples.gpt3_generation import Inference
from examples.bertshared_summarization import Summarize
from examples.mask_prediction import Predict


app = Flask(__name__)

print("model loading...")

# generate loading
generator = Inference('kykim/gpt3-kor-small_based_on_gpt2')

# summarize loading
summarizer = Summarize('kykim/bertshared-kor-base')

# Bert predict loading
predictor = Predict()


print("complete model loading")

requests_queue = Queue()    # request queue.
BATCH_SIZE = 1              # max request size.
CHECK_INTERVAL = 0.1


##
# Request handler.
# GPU app can process only one request in one time.
def handle_requests_by_batch():
    while True:
        request_batch = []

        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in request_batch:
                try:
                    types = requests['input'].pop(0)

                    if types == 'bert':
                        requests["output"] = run_predict(requests['input'][0], requests['input'][1])
                    elif types == 'albert':
                        requests["output"] = run_predict(requests['input'][0], requests['input'][1], types)
                    elif types == 'summarize':
                        requests["output"] = run_summarize(requests['input'][0], requests['input'][1])
                    elif types == 'generate':
                        requests["output"] = run_generate(requests['input'][0], requests['input'][1],
                                                          requests['input'][2])

                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


##
# predict with bert and albert model
def run_predict(text, samples=3, types='bert'):
    try:
        results = predictor.predict(text, types=types, top_clean=samples)

        return results

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


##
# bert shared text summarizer
def run_summarize(text, samples=1):
    try:
        results = summarizer(text, samples)

        return results

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


##
# GPT-3 generator.
def run_generate(text, samples=1, length=100):
    try:
        result = generator(text, samples, length)

        return result

    except Exception as e:
        print('Error occur in script generating!', e)
        return jsonify({'error': e}), 500


##
# Get post request page.
@app.route('/<types>', methods=['POST'])
def generate(types):
    if types not in ['bert', 'albert', 'summarize', 'gpt-3']:
        return jsonify({'Error': 'Not allow type'}), 404

    # GPU app can process only one request in one time.
    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'Error': 'Too Many Requests'}), 429

    try:
        print(types)

        args = []

        text = request.form['text']
        samples = int(request.form['samples'])

        args.append(types)
        args.append(text)
        args.append(samples)

        if types == 'generate':
            length = int(request.form['length'])
            args.append(length)

    except Exception as e:
        return jsonify({'message': 'Invalid request'}), 500

    # input a request on queue
    req = {'input': args}
    requests_queue.put(req)

    # wait
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return jsonify(req['output'])


##
# Queue deadlock error debug page.
@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return "Clear", 200


##
# Sever health checking page.
@app.route('/healthz', methods=["GET"])
def health_check():
    return "Health", 200


##
# Main page.
@app.route('/')
def main():
    return render_template('main.html'), 200


if __name__ == '__main__':
    from waitress import serve
    serve(app, port=80, host='0.0.0.0')
