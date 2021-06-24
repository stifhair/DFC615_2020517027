import json
import os
train_file = os.path.join(".", "KorQuAD_v1.0_train.json")
dev_file = os.path.join(".", "KorQuAD_v1.0_dev.json")

korquad_train = json.load(open(train_file,'r',encoding='utf-8'))
korquad_dev = json.load(open(dev_file,'r',encoding='utf-8'))
print(korquad_train.keys())

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from collections import Counter
import sys
import argparse
import string
import re
import glob
import logging
import random
import timeit

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)

from transformers import (
    ElectraForQuestionAnswering,
    XLMRobertaForQuestionAnswering,
    BertForQuestionAnswering,

    ElectraTokenizer,
    XLMRobertaTokenizer,

    ElectraConfig,
    XLMRobertaConfig
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from transformers import pipeline

logger = logging.getLogger(__name__)

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

MODEL_FOR_QUESTION_ANSWERING = {
    "koelectra-base-v3": ElectraForQuestionAnswering,
    "koelectra-small-v3": ElectraForQuestionAnswering,
}
TOKENIZER_CLASSES = {
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
}
CONFIG_CLASSES = {
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
}


def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'official_exact_match': exact_match, 'official_f1': f1}


def eval_during_train(args, step):
    expected_version = 'KorQuAD_v1.0'

    dataset_file = os.path.join(args.data_dir, args.predict_file)
    prediction_file = os.path.join(args.output_dir, 'predictions_{}.json'.format(step))

    with open(dataset_file) as dataset_f:
        dataset_json = json.load(dataset_f)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            print('Evaluation expects ' + expected_version +
                  ', but got dataset with ' + read_version,
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_f:
        predictions = json.load(prediction_f)

    return evaluate(dataset, predictions)


logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    # Added here for reproductibility
    set_seed(args)

    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "distilkobert", "xlm-roberta"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:
                        results = evaluation(args, model, tokenizer, global_step=global_step)
                        for key in sorted(results.keys()):
                            logger.info("  %s = %s", key, str(results[key]))

                    logging_loss = tr_loss

                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluation(args, model, tokenizer, global_step=None):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "distilkobert", "xlm-roberta"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            outputs = model(**inputs)

            start_logits = outputs[0]  # bs 512
            end_logits = outputs[1]  # torch.Tensor

            # print(len(start_logits)
            # print(len(start_logits[0]))

        # print(len(end_logits))
        # print(len(end_logits[0]))

        for i, example_index in enumerate(example_indices):
            start_logit = start_logits[i].cpu().tolist()
            end_logit = end_logits[i].cpu().tolist()
            # start_logits = start_logits[i]
            # end_logits = end_logits[i]
            # start_logits = start_logits[i]
            # end_logits = end_logits[i]
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = SquadResult(unique_id, start_logit, end_logit)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(global_step))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(global_step))
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    # Write the result
    # Write the evaluation result on file
    output_dir = os.path.join(args.output_dir, 'eval')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "eval_result_{}_{}.txt".format(
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        global_step))

    with open(output_eval_file, "w", encoding='utf-8') as f:
        official_eval_results = eval_during_train(args, step=global_step)
        results.update(official_eval_results)

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(os.path.join(args.data_dir),
                                                      filename=args.predict_file)
            else:
                examples = processor.get_train_examples(os.path.join(args.data_dir),
                                                        filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main(args):
    # Read from config file and make args
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.output_dir)

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    init_logger()
    set_seed(args)

    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

    # Load pretrained model and tokenizer
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
    )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce model loading logs
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_QUESTION_ANSWERING[args.model_type].from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluation(args, model, tokenizer, global_step=global_step)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

if __name__ == "__main__":

## 명령행 인터페이스, 사전 훈련된 모델을 로드하여 훈련하고 평가하는데까지 필요한 인자를 사용자 정의 인자로 설정하는 내용을 담고 있음.
## 사용법: python example.py --dataset_name squad.json ....
## 주피터 환경을 위한 인자 설정법
# "output_dir": "/content/gdrive/My Drive/Colab Notebooks/koelectra-small-korquad-ckpt",

    os.chdir('.')
    data_dir = "."
    output_dir = os.path.join(".", "res")
    # train_file = os.path.join(gdrive_path, "KorQuAD_v1.0_train.json")
    # dev_file = os.path.join(gdrive_path, "KorQuAD_v1.0_dev.json")

    hyper_para={
    "task": "korquad",
    "data_dir": data_dir,
    "ckpt_dir": "ckpt",
    "train_file": "KorQuAD_v1.0_train.json",
    "predict_file": "KorQuAD_v1.0_dev.json",
    "threads": 4,
    "version_2_with_negative": False,
    "null_score_diff_threshold": 0.0,
    "max_seq_length": 512,
    "doc_stride": 128,
    "max_query_length": 64,
    "max_answer_length": 30,
    "n_best_size": 20,
    "verbose_logging": True,
    "overwrite_output_dir": True,
    "evaluate_during_training": True,
    "eval_all_checkpoints": True,
    "save_optimizer": False,
    "do_lower_case": False,
    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 1,
    "weight_decay": 0.0,
    "gradient_accumulation_steps": 1,
    "adam_epsilon": 1e-8,
    "warmup_proportion": 0,
    "max_steps": -1,
    "max_grad_norm": 1.0,
    "no_cuda": False,
    "model_type": "koelectra-small-v3",
    "model_name_or_path": "monologg/koelectra-small-v3-discriminator",
    "output_dir": output_dir,
    "seed": 42,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "logging_steps": 3000,
    "save_steps": 3000,
    "learning_rate": 5e-5
    }

    main(AttrDict(hyper_para))

    test_file = os.path.join(gdrive_path, 'test.json')
    test_json = json.load(open(test_file, 'r', encoding='utf-8'))

    for sample in test_json['data']:
        context = sample['context']
        question = sample['question']
        print(f'context {context}')
        print(f'question {question}')
        print('\n')

    tokenizer = ElectraTokenizer.from_pretrained("./res/checkpoint-3000")
    model = ElectraForQuestionAnswering.from_pretrained("./res/checkpoint-3000")
    prediction_qa = pipeline("question-answering", tokenizer=tokenizer, model=model)

    context_question = {
        'question': '도지코인은 누구의 영향을 받나요?',
        'context': '일론머스크의 한마디에 도지코인은 오르락 내리락 오늘도 그의 트위터를 기다린다. 빨간맛 가즈아'

    }
    answer = prediction_qa(context_question)
    print(answer)



