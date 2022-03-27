import torch
import torch.nn as nn
import numpy as np

from utils import rank_array
from candidate_sampling import candidate_subsampling
from model_moe import MoE, MLPTower, MLPExpert



class ModelMultitaskBinaryTail(nn.Module):

    def __init__(self, pretrained_model, tokenizer, args):
        super(ModelMultitaskBinaryTail, self).__init__()
        self.tokenizer = tokenizer
        self.args = args

        # LM
        self.pretrained_model = pretrained_model
        # shared bottom
        self.fc1 = nn.Linear(2 * args.hidden_size, args.bottom_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.bottom_hidden_size, args.hidden_size)
        # MoE
        self.moe = MoE(args.device, args.n_tasks, args.hidden_size, args.hidden_size, args.num_experts, args.expert_hidden_size, args.k)
        # towers - one for each task
        self.towers = nn.ModuleList([MLPTower(args.hidden_size, args.tower_hidden_size) for i in range(args.n_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, mode, text_ids, text_mask, text_and_summaries_ids, text_and_summaries_mask, scores):
        loss = torch.tensor(0.0).to(self.pretrained_model.device)
        accuracy = [0 for j in range(self.args.n_tasks)]
        rank = [0 for j in range(self.args.n_tasks)]
        predictions = [[] for j in range(self.args.n_tasks)]
        for i in range(text_and_summaries_ids.shape[0]):

            # data
            text_and_summaries_ids_i = text_and_summaries_ids[i]
            text_and_summaries_mask_i = text_and_summaries_mask[i]

            # labels construction
            scores_i = scores[i]
            labels_i = torch.zeros(scores_i.shape, device = self.pretrained_model.device)
            for j in range(self.args.n_tasks):
                best_j = scores_i[j].max()
                if self.args.sharp_pos:
                    if best_j > scores_i[j].min():
                        labels_i[j][scores_i[j] == best_j] = 1
                else:
                    labels_i[j][scores_i[j] == best_j] = 1
            original_labels_i = labels_i.clone().detach()

            # candidate sampling
            selected_idx, text_and_summaries_ids_i, text_and_summaries_mask_i, scores_i, labels_i = candidate_subsampling(
                mode, text_and_summaries_ids_i, text_and_summaries_mask_i, scores_i, labels_i, self.args
            )

            # model output
            # LM encoding
            outputs_i = self.pretrained_model(
                input_ids=text_and_summaries_ids_i, attention_mask=text_and_summaries_mask_i, output_hidden_states=True
            )
            encs = outputs_i["last_hidden_state"]
            encs = encs[:, 0, :]
            outputs_tail_i = self.pretrained_model(
                input_ids=text_tail_and_summaries_ids_i, attention_mask=text_tail_and_summaries_mask_i, output_hidden_states=True
            )
            encs_tail = outputs_tail_i["last_hidden_state"]
            encs_tail= encs_tail[:, 0, :]
            encs = torch.cat((encs, encs_tail), -1)
            # shared bottom
            if self.args.use_shared_bottom:
                preds_i = self.fc2(self.relu(self.fc1(encs)))
            else:
                preds_i = encs
            # MoE
            train = torch.sum(mode) > 0
            preds_i, aux_loss_i = self.moe(preds_i, train=train, collect_gates=not (train))

            loss_i = torch.tensor(0.0).to(self.pretrained_model.device)
            for j in range(self.args.n_tasks):
                # pred
                preds_i_j = self.towers[j](preds_i[j])[:, 0]

                # labels
                labels_i_j = torch.zeros(len(preds_i_j), device=torch.device("cuda"))
                pos_idx = scores_i[j].argmax().item()
                labels_i_j[pos_idx] = 1

                # loss
                loss_i_j = self.loss(preds_i_j, labels_i_j)
                loss_i = loss_i + loss_i_j

                # predictions
                preds_i_j = self.sigmoid(preds_i_j).detach().cpu().numpy()
                prediction_idx = np.argmax(preds_i_j)

                # accuracy
                accuracy_i_j = 100 * int(scores_i[j][prediction_idx].item() == scores_i[j][pos_idx].item())
                accuracy[j] = accuracy[j] + accuracy_i_j

                # ranks
                ranks = rank_array(preds_i_j)
                all_pos_idx = [k for k in range(len(scores_i[j])) if
                               scores_i[j][k].item() == scores_i[j][pos_idx].item()]
                rank_i_j = np.min(ranks[all_pos_idx])
                rank[j] = rank[j] + rank_i_j

                # prediction
                prediction = scores_i[j][prediction_idx].item()
                predictions[j].append(prediction)
            loss_i = loss_i / self.args.n_tasks
            if self.args.use_aux_loss:
                loss_i = loss_i + aux_loss_i
            loss = loss + loss_i

        loss /= scores.shape[0]
        #print(loss)
        outputs = {
            "loss": loss,
            "loss_nce": loss,
        }
        prediction_sum = 0
        for j in range(self.args.n_tasks):
            accuracy[j] /= scores.shape[0]
            outputs["accuracy_{}".format(self.args.scoring_methods[j])] = torch.tensor(accuracy[j]).float().to(
                loss.device)
            rank[j] /= scores.shape[0]
            outputs["rank_{}".format(self.args.scoring_methods[j])] = torch.tensor(rank[j]).float().to(loss.device)
            predictions[j] = np.mean(predictions[j])
            outputs["prediction_{}".format(self.args.scoring_methods[j])] = torch.tensor(predictions[j]).float().to(
                loss.device)
            prediction_sum += predictions[j]
        outputs["prediction_sum"] = torch.tensor(prediction_sum).float().to(loss.device)
        print(prediction_sum)

        return outputs


class ModelMultitaskBinaryv2(nn.Module):

    def __init__(self, pretrained_model, tokenizer, args):
        super(ModelMultitaskBinaryv2, self).__init__()
        self.tokenizer = tokenizer
        self.args = args

        # LM
        self.pretrained_model = pretrained_model
        # shared bottom
        self.fc1 = nn.Linear(2 * args.hidden_size, args.bottom_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.bottom_hidden_size, args.hidden_size)
        # MoE
        self.moe = MoE(args.device, args.n_tasks, args.hidden_size, args.hidden_size, args.num_experts,
                       args.expert_hidden_size, args.k)
        # towers - one for each task
        self.towers = nn.ModuleList([MLPTower(args.hidden_size, args.tower_hidden_size) for i in range(args.n_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, mode, text_ids, text_mask, text_and_summaries_ids, text_and_summaries_mask, text_tail_and_summaries_ids, text_tail_and_summaries_mask, scores):
        loss = torch.tensor(0.0).to(self.pretrained_model.device)
        accuracy = [0 for j in range(self.args.n_tasks)]
        rank = [0 for j in range(self.args.n_tasks)]
        predictions = [[] for j in range(self.args.n_tasks)]
        total_predictions_idx = []
        for i in range(text_and_summaries_ids.shape[0]):

            # data
            text_ids_i = text_ids[i]
            text_mask_i = text_mask[i]
            text_and_summaries_ids_i = text_and_summaries_ids[i]
            text_and_summaries_mask_i = text_and_summaries_mask[i]
            scores_i = scores[i]
            if torch.sum(mode) > 0 and self.args.filter_out_duplicates:
                idx = unique_idx(scores[i])
                text_and_summaries_ids_i = text_and_summaries_ids_i[idx]
                text_and_summaries_mask_i = text_and_summaries_mask_i[idx]
                scores_i = scores_i[:, idx]
            if torch.sum(mode) > 0 and self.args.prune_candidates:
                idx_to_keep = prune_idx(scores_i, self.args)
                if self.args.pos_neg_construction == "per_task":
                    idx_to_keep = idx_to_keep[:self.args.max_n_candidates]
                text_and_summaries_ids_i = text_and_summaries_ids_i[idx_to_keep]
                text_and_summaries_mask_i = text_and_summaries_mask_i[idx_to_keep]
                scores_i = scores_i[:, idx_to_keep]

            # model output
            # LM encoding
            # cross-attention
            outputs_i = self.pretrained_model(
                input_ids=text_and_summaries_ids_i, attention_mask=text_and_summaries_mask_i, output_hidden_states=True
            )
            encs = outputs_i["last_hidden_state"]
            encs = encs[:, 0, :]
            # text
            outputs_text = self.pretrained_model(
                input_ids=text_ids_i, attention_mask=text_mask_i, output_hidden_states=True
            )
            encs_text = outputs_text["last_hidden_state"]
            encs_text = encs_text[:, 0, :]
            encs_text = encs_text.repeat((len(encs), 1))
            # shared bottom
            encs = torch.cat((encs, encs_text), 1)
            preds_i = self.fc2(self.relu(self.fc1(encs)))
            # MoE
            train = torch.sum(mode) > 0
            preds_i, aux_loss_i = self.moe(preds_i, train=train, collect_gates=not (train))

            loss_i = torch.tensor(0.0).to(self.pretrained_model.device)
            total_predictions = np.zeros(len(preds_i[0]))
            for j in range(self.args.n_tasks):
                # pred
                preds_i_j = self.towers[j](preds_i[j])[:, 0]

                # labels
                labels_i_j = torch.zeros(len(preds_i_j), device=torch.device("cuda"))
                pos_idx = scores_i[j].argmax().item()
                labels_i_j[pos_idx] = 1

                # loss
                loss_i_j = self.loss(preds_i_j, labels_i_j)
                loss_i = loss_i + loss_i_j

                # predictions
                preds_i_j = self.sigmoid(preds_i_j).detach().cpu().numpy()
                prediction_idx = np.argmax(preds_i_j)

                # accuracy
                accuracy_i_j = 100 * int(scores_i[j][prediction_idx].item() == scores_i[j][pos_idx].item())
                accuracy[j] = accuracy[j] + accuracy_i_j

                # ranks
                ranks = rank_array(preds_i_j)
                all_pos_idx = [k for k in range(len(scores_i[j])) if
                               scores_i[j][k].item() == scores_i[j][pos_idx].item()]
                rank_i_j = np.min(ranks[all_pos_idx])
                rank[j] = rank[j] + rank_i_j

                # prediction
                prediction = scores_i[j][prediction_idx].item()
                predictions[j].append(prediction)

                total_predictions += preds_i_j
            loss_i = loss_i / self.args.n_tasks
            if self.args.use_aux_loss:
                loss_i = loss_i + aux_loss_i
            loss = loss + loss_i

            total_prediction_idx = np.argmax(total_predictions)
            total_predictions_idx.append(total_prediction_idx)

        loss /= scores.shape[0]
        outputs = {
            "loss": loss,
            "loss_nce": loss,
            "total_predictions_idx": total_predictions_idx
        }
        prediction_sum = 0
        for j in range(self.args.n_tasks):
            accuracy[j] /= scores.shape[0]
            outputs["accuracy_{}".format(self.args.scoring_methods[j])] = torch.tensor(accuracy[j]).float().to(
                loss.device)
            rank[j] /= scores.shape[0]
            outputs["rank_{}".format(self.args.scoring_methods[j])] = torch.tensor(rank[j]).float().to(loss.device)
            predictions[j] = np.mean(predictions[j])
            outputs["prediction_{}".format(self.args.scoring_methods[j])] = torch.tensor(predictions[j]).float().to(
                loss.device)
            prediction_sum += predictions[j]
        outputs["prediction_sum"] = torch.tensor(prediction_sum).float().to(loss.device)

        return outputs
