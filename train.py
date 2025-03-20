import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
import torch
from torch.nn import Sigmoid, Softmax
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage, CharFbeta, Accuracy
from eval_utils import Recall, Precision
from ignite.metrics import Bleu, RougeL, RougeN
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, WEIGHTS_NAME, CONFIG_NAME)
from utils_focus import make_focus_logdir
from data_utils import get_data_loaders, add_special_tokens_, load_idf

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def train():
    parser = ArgumentParser()
    parser.add_argument("--kp_method", type=str, default="comac", help="{focus, comac}")
    parser.add_argument("--model_name", type=str, default="", help="{GPT2, BART}")
    parser.add_argument("--gpt2_model_path", type=str, default="gpt2",
                        help="pre-trained model path for decoder only models")  # gpt2-medium
    parser.add_argument("--bart_model_path", type=str, default="facebook/bart-base",
                        help="pre-trained model path for encoder-decoder models")  # facebook/bart-large
    parser.add_argument("--train_dataset_path", type=str, default="data/train_focus.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--train_dataset_cache", type=str, default='data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--dev_dataset_path", type=str, default="data/valid_focus.json",
                        help="Path or url of the dataset.")
    parser.add_argument("--dev_dataset_cache", type=str, default='data/focus_cache.tar.gz',
                        help="Path or url of the dataset cache")
    parser.add_argument("--ps_coef", type=float, default=1.0, help="Coefficient for persona loss")
    parser.add_argument("--kn_coef", type=float, default=1.0, help="Coefficient for knowledge loss")
    parser.add_argument("--lm_coef", type=float, default=10.0, help="Coefficient for LM loss")
    parser.add_argument("--max_history", type=int, default=1, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=1, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")
    parser.add_argument("--inference", action='store_true', help="If true, inference with gold knowledge")
    parser.add_argument("--test_infer", action='store_true', help="If true, test inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--gpu_start_num", type=int, default=1, help="Start number of GPU")
    parser.add_argument("--flag", type=str, default="", help="Assign the name of the folder")
    parser.add_argument("--model_dir", type=str, default="models", help="Parent folder for storing the models.")
    parser.add_argument("--seed", type=int, default=19950604)
    parser.add_argument("--random_knowledge", action='store_true',
                        help="If true, the model choose the knowledge randomly")
    parser.add_argument("--incontext", action='store_true', help="If true, it will use incontext structure")
    parser.add_argument("--pg_label_weight", type=float, required=False, help="Weight on positive PG label in loss.")
    parser.add_argument("--pg_loss_sample_p", type=float, required=False,
                        help="Probability of keeping PG training example if all labels are negative in PG loss (only affect PG loss).")
    parser.add_argument("--sncolbert_sample_rate", type=float, default=0.35,
                        help="Rate for sampling tokens in SNColBERT layer.")
    parser.add_argument("--idf_file", type=str, required=False,
                        help="Term idf (inverse document frequency) file used for SNColBERT token sampling")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Arguments: %s", pformat(args))

    args.distributed = (args.local_rank != -1)
    if args.distributed:
        local_rank = args.local_rank + args.gpu_start_num
        print("args local rank: ", args.local_rank, " local rank: ", local_rank)
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")

    if args.model_name == 'GPT2':
        from transformers import GPT2Tokenizer
        if args.kp_method == 'focus':
            from classification_modules import GPT2_focus as gpt2model
        elif args.kp_method == 'comac':
            from classification_modules import GPT2_comac as gpt2model
        else:
            raise ValueError(f'Unknown kp_method: {args.kp_method}')
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_path)
        model = gpt2model.from_pretrained(args.gpt2_model_path)
        model.to(args.device)
        model.eval()
        if args.gpt2_model_path == 'gpt2' or 'gpt2-medium':
            add_special_tokens_(model, tokenizer)

    elif args.model_name == 'BART':
        from transformers import BartTokenizer
        if args.kp_method == 'focus':
            from classification_modules import BART_focus as bartmodel
        elif args.kp_method == 'comac':
            from classification_modules import BART_comac as bartmodel
        else:
            raise ValueError(f'Unknown kp_method: {args.kp_method}')
        tokenizer = BartTokenizer.from_pretrained(args.bart_model_path)
        model = bartmodel.from_pretrained(args.bart_model_path)
        model.to(args.device)
        model.eval()
        if args.bart_model_path == "facebook/bart-base" or "facebook/bart-large":
            add_special_tokens_(model, tokenizer)

    else:
        raise NotImplementedError

    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # load term frequency file for SNColBERT layer
    if args.kp_method == 'comac':
        idf = load_idf(args.idf_file)
        model.set_idf(idf, args.device)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        if model.config.model_type == 'gpt2':
            input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = batch
            output = model(
                input_ids=input_ids,
                input_eos=input_eos,
                token_type_ids=token_type_ids,
                only_dial_input_ids=dialog,
                only_dial_token_type_ids=dialog_tti,
                persona_input_ids=persona_candidates,
                knowledge_input_ids=knowledge_candidates,
                persona_can_idx=persona_can_idx,
                persona_grounding=persona_grounding,
                knowledge_can_idx=knowledge_can_idx,
                knowledge_grounding=knowledge_grounding,
                tot_knowledge=tot_knowledge,
                tot_knowledge_token_ids=tot_knowledge_token_ids,
                tot_knowledge_eos=tot_knowledge_eos,
                training=True,
                mc_token_ids=mc_token_ids,
                pg_label_weight=args.pg_label_weight,
                pg_loss_sample_p=args.pg_loss_sample_p,
                sncolbert_sample_rate=args.sncolbert_sample_rate
            )
        elif model.config.model_type == 'bart':
            input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = batch
            output = model(
                input_ids=input_ids,
                input_eos=input_eos,
                only_dial_input_ids=dialog,
                decoder_input_ids=decoder_input_ids,
                persona_input_ids=persona_candidates,
                knowledge_input_ids=knowledge_candidates,
                persona_can_idx=persona_can_idx,
                persona_grounding=persona_grounding,
                knowledge_can_idx=knowledge_can_idx,
                knowledge_grounding=knowledge_grounding,
                tot_knowledge=tot_knowledge,
                tot_knowledge_eos=tot_knowledge_eos,
                lm_labels=lm_labels,
                training=True,
                mc_token_ids=mc_token_ids,
                pg_label_weight=args.pg_label_weight,
                pg_loss_sample_p=args.pg_loss_sample_p,
                sncolbert_sample_rate=args.sncolbert_sample_rate
            )
        else:
            raise NotImplementedError
        # train: lm_loss, knowledge_loss, persona_loss, dynamic_lm_logits, knowledge_logits, persona_logits
        # valid: lm_label, dynamic_lm_logits, knowledge_logits, persona_logits
        lm_loss, knowledge_loss, persona_loss = output[0], output[1], output[2]
        loss = (
                           lm_loss * args.lm_coef + knowledge_loss * args.kn_coef + persona_loss * args.ps_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return (lm_loss.item(), knowledge_loss.item(), persona_loss.item())

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            if model.config.model_type == 'gpt2':
                input_ids, input_eos, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                    knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_token_ids, tot_knowledge_eos, reply, dialog, dialog_tti = batch

                output = model(
                    input_ids=input_ids,
                    input_eos=input_eos,
                    token_type_ids=token_type_ids,
                    only_dial_input_ids=dialog,
                    only_dial_token_type_ids=dialog_tti,
                    persona_input_ids=persona_candidates,
                    knowledge_input_ids=knowledge_candidates,
                    persona_can_idx=persona_can_idx,
                    knowledge_can_idx=knowledge_can_idx,
                    tot_knowledge=tot_knowledge,
                    tot_knowledge_token_ids=tot_knowledge_token_ids,
                    tot_knowledge_eos=tot_knowledge_eos,
                    training=False,
                    mc_token_ids=mc_token_ids,
                    pg_label_weight=args.pg_label_weight,
                    pg_loss_sample_p=args.pg_loss_sample_p,
                    sncolbert_sample_rate=args.sncolbert_sample_rate
                )
                # train: lm_loss, knowledge_loss, persona_loss, dynamic_lm_logits, knowledge_logits, persona_logits
                # valid: lm_label, dynamic_lm_logits, knowledge_logits, persona_logits
                lm_labels, lm_logits, knowledge_logits, persona_logits = output[0], output[1], output[2], output[3]


            elif model.config.model_type == 'bart':
                input_ids, input_eos, decoder_input_ids, lm_labels, token_type_ids, mc_token_ids, persona_candidates, persona_can_idx, persona_grounding, knowledge_candidates, \
                    knowledge_can_idx, knowledge_grounding, tot_knowledge, tot_knowledge_eos, reply, dialog = batch
                output = model(
                    input_ids=input_ids,
                    input_eos=input_eos,
                    only_dial_input_ids=dialog,
                    decoder_input_ids=decoder_input_ids,
                    persona_input_ids=persona_candidates,
                    knowledge_input_ids=knowledge_candidates,
                    persona_can_idx=persona_can_idx,
                    knowledge_can_idx=knowledge_can_idx,
                    tot_knowledge=tot_knowledge,
                    tot_knowledge_eos=tot_knowledge_eos,
                    training=False,
                    mc_token_ids=mc_token_ids,
                    pg_label_weight=args.pg_label_weight,
                    pg_loss_sample_p=args.pg_loss_sample_p,
                    sncolbert_sample_rate=args.sncolbert_sample_rate
                )
                lm_logits, knowledge_logits, persona_logits = output[0], output[1], output[2]


            else:
                raise NotImplementedError

            lm_logits_flat_shifted = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[:, 1:].contiguous().view(-1)

            persona_logits = persona_logits.squeeze()
            persona_grounding = persona_grounding.type_as(persona_logits).squeeze()

            sigmoid = Sigmoid()
            persona_pred_sigmoid = sigmoid(persona_logits)
            persona_pred_sigmoid = (persona_pred_sigmoid > 0.5).float()

            softmax = Softmax(dim=-1)
            knowledge_pred = softmax(knowledge_logits)
            _, k_index_1 = torch.topk(knowledge_pred, k=1, dim=-1)
            _, k_index_5 = torch.topk(knowledge_pred, k=5, dim=-1)
            k_index_1, k_index_5 = k_index_1.squeeze(0), k_index_5.squeeze(0)
            k_index_1_cvtd = torch.tensor([1 if num in k_index_1 else 0 for num in range(10)], device=args.device)
            k_label_cvtd = torch.tensor([1 if num in knowledge_grounding else 0 for num in range(10)],
                                        device=args.device)

            lm_pred = softmax(lm_logits_flat_shifted)
            lm_val, lm_idx = torch.topk(lm_pred, k=1, dim=-1)
            lm_idx = lm_idx.squeeze(-1)

            mask = (lm_labels_flat_shifted != -100)
            lm_labels_only = [lm_labels_flat_shifted[mask].tolist()]
            lm_idx_only = lm_idx[mask].tolist()

            return (lm_logits_flat_shifted, knowledge_logits, persona_logits, persona_pred_sigmoid, k_index_1_cvtd,
                    knowledge_pred, lm_idx_only), \
                (lm_labels_flat_shifted, knowledge_grounding, persona_grounding.type_as(persona_logits), k_label_cvtd,
                 lm_labels_only)

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "lm_loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "knowledge_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "persona_loss")

    metrics = {
        "lm_loss": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
        "knowledge_loss": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0][1], x[1][1])),
        "persona_loss": Loss(torch.nn.BCEWithLogitsLoss(), output_transform=lambda x: (x[0][2], x[1][2])),
        "Knowledge_acc": Accuracy(output_transform=lambda x: (x[0][5], x[1][1])),
        "Knowledge_pr": Precision(output_transform=lambda x: (x[0][4], x[1][3])),
        "Knowledge_rc": Recall(output_transform=lambda x: (x[0][4], x[1][3])),
        "Persona_acc": Accuracy(output_transform=lambda x: (x[0][3], x[1][2])),
        "Persona_pr": Precision(output_transform=lambda x: (x[0][3], x[1][2])),
        "Persona_rc": Recall(output_transform=lambda x: (x[0][3], x[1][2])),
        "Persona_acc_mtl": Accuracy(output_transform=lambda x: (x[0][3].unsqueeze(0), x[1][2].unsqueeze(0)),
                                    is_multilabel=True)
    }

    metrics.update({"average_lm_loss": MetricsLambda(average_distributed_scalar, metrics["lm_loss"], args),
                    "average_knowledge_loss": MetricsLambda(average_distributed_scalar, metrics["knowledge_loss"],
                                                            args),
                    "average_persona_loss": MetricsLambda(average_distributed_scalar, metrics["persona_loss"], args),
                    "average_Knowledge_acc": MetricsLambda(average_distributed_scalar, metrics["Knowledge_acc"], args),
                    "average_Knowledge_pr": MetricsLambda(average_distributed_scalar, metrics["Knowledge_pr"], args),
                    "average_Knowledge_rc": MetricsLambda(average_distributed_scalar, metrics["Knowledge_rc"], args),
                    "average_Persona_acc": MetricsLambda(average_distributed_scalar, metrics["Persona_acc"], args),
                    "average_Persona_pr": MetricsLambda(average_distributed_scalar, metrics["Persona_pr"], args),
                    "average_Persona_rc": MetricsLambda(average_distributed_scalar, metrics["Persona_rc"], args),
                    "average_Persona_acc_mtl": MetricsLambda(average_distributed_scalar, metrics["Persona_acc_mtl"],
                                                             args)
                    })

    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_lm_loss"])

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["lm_loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        model_dir = args.model_dir
        # model_dir_identigier = str(os.path.basename(__file__))[:-3] + "_" + args.model_name + "_" + args.flag
        model_dir_identigier = args.flag
        log_dir = make_focus_logdir(model_dir, model_dir_identigier)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training",
                                                            metric_names=["lm_loss", "knowledge_loss", "persona_loss",
                                                                          "knowledge_accuracy", "persona_accuracy",
                                                                          "f1_score"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.EPOCH_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys())),
                         event_name=Events.ITERATION_COMPLETED(every=5000))
        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))
        tb_logger.close()


if __name__ == "__main__":
    import time

    start = time.time()
    train()
    end = time.time()
    print("Total time needed:", end - start)
