import argparse
import bert_score
import json
import logging
import os
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, OPTForCausalLM




def init_scorer():
    transformers.tokenization_utils.logger.setLevel(logging.ERROR)
    transformers.configuration_utils.logger.setLevel(logging.ERROR)
    transformers.modeling_utils.logger.setLevel(logging.ERROR)
    from bert_score import BERTScorer
    #scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge", lang='en', rescale_with_baseline=True)
    return scorer


def load_bert_scorer():
    scorer = init_scorer()
    scorer.score(['Patient has silicosis'], ['Patient has tuberculosis'])
    return

def get_answer(output_buf):
    out_str = str(output_buf)
    query_str = 'Answer: '
    pos = out_str.find(query_str)
    answer = out_str[pos + len(query_str):]

    # Scrub the answer string
    pos_1 = answer.find('\\')
    pos_2 = answer.find('\</s')
    end_pos = pos_1 if pos_1 < pos_2 else pos_2
    return answer[:end_pos]

def prompt_and_compare(model, tokenizer, prompt, max_tokens, ref_answer, scorer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    beam_outputs = model.generate(input_ids, num_beams=5, do_sample=True,
                num_return_sequences=5, max_new_tokens=max_tokens)

    answer_list = []
    for i, beam_output in enumerate(beam_outputs):
        answer = "{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True))
        print(f'ANSWER ({i+1}):')
        print(answer)
        cmp_score = scorer.score([answer], [ref_answer])
        print(f'SCORE: {cmp_score}')
        answer_list.append((cmp_score, answer))


    answers_sorted_by_score = sorted(answer_list, key=lambda tup: tup[0])
    return answers_sorted_by_score[0]
    #output_buf = []
   #
    #for tok in outputs:
        #print(transformers_tokenizer.decode(outputs[0]))
        #output_buf.append(tokenizer.decode(tok))
    #print(output_buf)
    #answer = get_answer(output_buf)
    #cmp_score = scorer.score([answer], [ref_answer])
    #return answer, cmp_score
    
def exec_prompt(model, tokenizer, prompt, max_tokens):
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
  outputs = model.generate(input_ids, num_beams=5, do_sample=True, max_new_tokens=max_tokens)
  output_buf = []
  for tok in outputs:
    #print(transformers_tokenizer.decode(outputs[0]))
    output_buf.append(tokenizer.decode(tok))
  print(output_buf)
  answer = get_answer(output_buf)
  return answer

def init_models(model_name='facebook/galactica-30b'):
  model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  return model, tokenizer

def run_eval(args):
  transformers_model = OPTForCausalLM.from_pretrained(args.model, \
      torch_dtype=torch.float16, device_map="auto")
  transformers_tokenizer = AutoTokenizer.from_pretrained(args.model)

  if args.compare:
    print('Comparison will be done via BERTScorer')
    scorer = init_scorer()
  else:
	print('Comparison option is disabled.')

  with open(args.query_path) as f:
    question_list = json.load(f)

  print(f'Processing {len(question_list)} questions from: {args.query_path}')
  query_results = []

  for q in question_list:
    result = {'query': q['question']}
    
if args.compare:
      if args.prompt_work:
        prompt = 'Question: ' + q['question'] + '\n\nAnswer:<work>'
      else:
        prompt = 'Question: ' + q['question'] + '\n\nAnswer:'
      answer, cmp_score = prompt_and_compare(transformers_model,
          transformers_tokenizer, prompt,
          args.max_num_tokens, q['answer'], scorer)
	  result['answer'] = answer
      result['cmp_score'] = cmp_score
    else:
      if args.prompt_work:
        prompt = 'Question: ' + q['question'] + '\n\nAnswer:<work>'
      else:
        prompt = 'Question: ' + q['question'] + '\n\nAnswer:'
      answer = exec_prompt(transformers_model, transformers_tokenizer, prompt,
          args.max_num_tokens)
      result['answer'] = answer

    query_results.append(result)

  query_name = os.path.basename(args.query_path).replace('.json', '')
  model_name = args.model.replace('/', '_')
  if args.job_name != '':
    out_path = f'{args.out_dir}/{args.job_name}_{model_name}_{query_name}.csv'
  else:
	  out_path = f'{args.out_dir}/{model_name}_{query_name}.csv'
  pd.DataFrame(query_results).to_csv(out_path, index=False)
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--max_num_tokens",
    type=int,
    help="Specify the maximum number of tokens in generated output"
  )
  parser.add_argument(
    "--query_path",
    type=str,
    help="Specify path of the JSON file containing a list of queries"
  )
  parser.add_argument(
    "--model",
    type=str,
    default="facebook/galactica-30b",
    help="Specify language model"
  )
  parser.add_argument(
    "--out_dir",
    type=str,
    default=".",
    help="Specify output directory to save results"
  )
  parser.add_argument(
    "--job_name",
    type=str,
    default="",
    help="Specify a job name to save an unique log"
  )
   parser.add_argument(
    "--prompt_work",
    type=bool,
    default=False,
    help="Indicate if the <work> prompt should be used"
  )
  parser.add_argument(
    "--compare",
    type=bool,
    default=False,
    help="Indicate if the answer should be compared with a reference"
  )
  args = parser.parse_args()
  print('----------------------------------------------')
  print('Command line arguments:            ')
  print('----------------------------------------------')
  print(args)
  print('----------------------------------------------')
  run_eval(args)
